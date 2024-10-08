import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import grad, Variable
import einops

import numpy as np
import matplotlib.pyplot as plt
import copy
import dill as pickle
import os
import time


import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import h5py as h5
import hdf5plugin
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc, accuracy_score

import contextlib
import warnings

def SO3_gens():
    Lz = torch.tensor([[0,1,0],[-1,0,0],[0,0,0]],dtype = torch.float32)
    Ly = torch.tensor([[0,0,-1],[0,0,0],[1,0,0]],dtype = torch.float32)
    Lx = torch.tensor([[0,0,0],[0,0,1],[0,-1,0]],dtype = torch.float32)
    gens = [Lx, Ly, Lz]
    return gens


def Lorentz_gens():

    #p^a= p[a] -> Jz[a,b]p[b] (p_a = p[a] -> Jz[a,b]p[b])
    Lz = torch.tensor([[0,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,0]],dtype = torch.float32)
    Ly = torch.tensor([[0,0,0,0],[0,0,0,-1],[0,0,0,0],[0,1,0,0]],dtype = torch.float32)
    Lx = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,-1,0]],dtype = torch.float32)


    #p^a = p[a] -> Kz[a,b]p[b] (p_a = p[a] -> -Kz[a,b]p[b])
    Kz = torch.tensor([[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]],dtype = torch.float32)
    Ky = torch.tensor([[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],dtype = torch.float32)
    Kx = torch.tensor([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]],dtype = torch.float32)

    gens = [Kx, Ky, Kz, Lx, Ly, Lz]
    return gens


gens_SO3 = einops.rearrange(SO3_gens(),'n h w -> n h w')


gens_Lorentz = einops.rearrange(Lorentz_gens(),'n h w -> n h w')


devicef = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {devicef} device")

def shuffle_fun(a ,b ,c):
    idx = np.random.permutation(len(a))
    return a[idx], b[idx], c[idx]


def to_cylindrical(four_vec, log=True):
    E = four_vec[:,:,0]
    px = four_vec[:,:,1]
    py = four_vec[:,:,2]
    pz = four_vec[:,:,3]
    pt = torch.sqrt(px*px + py*py)
    phi = torch.arctan2(py,px)
    eta = torch.arcsinh(pz/pt)

    if log:
        cylindrical_four_vec = torch.cat([
            torch.log(E.unsqueeze(-1)),
            torch.log(pt.unsqueeze(-1)), 
            eta.unsqueeze(-1),
            phi.unsqueeze(-1)
        ], axis=2)

        cylindrical_four_vec = torch.where(cylindrical_four_vec < -1e30, 0, cylindrical_four_vec)
    else:
        cylindrical_four_vec = torch.cat([E.unsqueeze(-1),pt.unsqueeze(-1), eta.unsqueeze(-1),phi.unsqueeze(-1)], axis=2)

    
    return torch.nan_to_num(cylindrical_four_vec)



def create_transformation_dict(cartesian_features):
    # Converts Cartesian coordinates to cylindrical and returns a transformation dict
    # cartesian_features shape [B, N, 4]
    logE, logPt, eta, phi = to_cylindrical(cartesian_features, log=True).unbind(dim=2)
    
    # Intermediate quantities, shape [B, N]
    sin_phi = torch.sin(phi) 
    cos_phi = torch.cos(phi) 
    sinh_eta = torch.sinh(eta)
    cosh_eta = torch.cosh(eta)
    lamb = torch.exp(logE - logPt)
    zeros = torch.zeros_like(logE)
    ones = torch.ones_like(logE)
    
    # Create and return the transformation dictionary
    # Each key corresponds to a different coordinate
    # Each value is a tensor of shape [B, N, 6]
    trans_dict = {}
    trans_dict['logE'] = torch.stack([zeros]*3 + [cos_phi/lamb, sin_phi/lamb, sinh_eta/lamb], dim=2)
    trans_dict['logPt'] = torch.stack([sin_phi*sinh_eta, -cos_phi*sinh_eta, zeros, lamb*cos_phi, lamb*sin_phi, zeros], dim=2)
    trans_dict['eta'] = torch.stack([-cosh_eta*sin_phi, cosh_eta*cos_phi, zeros, -lamb*cos_phi*sinh_eta/cosh_eta, -lamb*sin_phi*sinh_eta/cosh_eta, lamb/cosh_eta], dim=2)
    trans_dict['phi'] = torch.stack([cos_phi*sinh_eta, sin_phi*sinh_eta, -ones, -lamb*sin_phi, lamb*cos_phi, zeros], dim=2)
    return trans_dict


def symm_loss_scalar7(model, X, X_cartesian, jet_vec, mask=None, train=False, generators=None, take_mean = False):
    # generator should be a list of 6 boolian values, indicating which of the 6 Lorentz generators to use
     # Check if generators is a list of boolean values
    if generators is not None:
        if not isinstance(generators, list) or len(generators) != 6 or not all(isinstance(g, bool) for g in generators):
            warnings.warn("Invalid 'generators'. Expected a list of 6 boolean values. Continuing with 'generators=None'.")
            generators = None
    
    device = X.device
    
    model = model.to(device) # Probably not necessary, I assume the model is already on the correct device
    

    # Create transformation dict for particles and jets
    trans_dict = create_transformation_dict(X_cartesian)
    trans_dict_jet = create_transformation_dict(jet_vec)
    
    # Prepare input
    input = X.clone().detach().requires_grad_(True).to(device)
    model.eval()
    output = model(input, mask)  # [B]
    
    # Compute gradients
    grads, = torch.autograd.grad(
        outputs=output, 
        inputs=input, 
        grad_outputs=torch.ones_like(output, device=device), 
        create_graph=train
    )
    
    
    with contextlib.nullcontext() if train else torch.no_grad():
        # Build the variation tensor [B,6]
        var_tensor = torch.einsum('b n, b n k -> b k',grads[:,:,0], (trans_dict['eta'] - trans_dict_jet['eta']))
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,1], (trans_dict['phi'] - trans_dict_jet['phi']))
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,2], (trans_dict['logPt'] - trans_dict_jet['logPt']))
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,3], trans_dict['logPt'])
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,4], (trans_dict['logE'] - trans_dict_jet['logE']))
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,5], trans_dict['logE'])
        
        # delta R calculation (K_tensor)
        K_tensor = (input[:,:,0].unsqueeze(-1) * (trans_dict['eta'] - trans_dict_jet['eta']))
        K_tensor += (input[:,:,1].unsqueeze(-1) * (trans_dict['phi'] - trans_dict_jet['phi']))
        K_tensor /= input[:,:,6].unsqueeze(-1)+1e-10 # Avoid division by zero
        var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,6], K_tensor)


        # Apply generators mask if provided
        if generators is not None:
            generators_tensor = torch.tensor(generators, dtype=torch.bool, device=device).unsqueeze(0)  # [1, 6]
            var_tensor = torch.where(generators_tensor, var_tensor, torch.zeros_like(var_tensor))

        
        # Compute the loss
        loss = torch.norm(var_tensor, p=2, dim=1)**2
        
        if take_mean:
            loss = loss.mean()
        return loss
    
    
    
    
def symm_loss_scalar9(model, X, X_cartesian, mask=None, train=False, generators=None, take_mean = False, dict_vars = {"logE":0, "logPt":1, "eta":2, "phi":3, "log_R_E":4, "log_R_pt":5, "dEta":6, "dPhi":7, "dR":8}):
    # generator should be a list of 6 boolian values, indicating which of the 6 Lorentz generators to use
     # Check if generators is a list of boolean values
    if generators is not None:
        if not isinstance(generators, list) or len(generators) != 6 or not all(isinstance(g, bool) for g in generators):
            warnings.warn("Invalid 'generators'. Expected a list of 6 boolean values. Continuing with 'generators=None'.")
            generators = None
    
    device = X.device
    
    model = model.to(device) # Probably not necessary, I assume the model is already on the correct device
    
    
    # Prepare input
    input = X.clone().detach().requires_grad_(True).to(device)
    model.eval()
    output = model(input, mask)  # [B]
    
    jet_vec = torch.sum(X_cartesian, dim=1).unsqueeze(1)
    
    # Create transformation dict for particles and jets
    trans_dict = create_transformation_dict(X_cartesian)
    trans_dict_jet = create_transformation_dict(jet_vec)
    
    dici = trans_dict
    dici["dEta"] = trans_dict_jet['eta'] - trans_dict['eta']
    dici["dPhi"] = trans_dict_jet['phi'] - trans_dict['phi']
    dici["log_R_pt"] = (trans_dict['logPt'] - trans_dict_jet['logPt'])
    dici["log_R_E"] = (trans_dict['logE'] - trans_dict_jet['logE'])
    K_tensor = (input[:,:,dict_vars["dEta"]].unsqueeze(-1) * (trans_dict['eta'] - trans_dict_jet['eta']))
    K_tensor += (input[:,:,dict_vars["dPhi"]].unsqueeze(-1) * (trans_dict['phi'] - trans_dict_jet['phi']))
    K_tensor /= input[:,:,dict_vars["dR"]].unsqueeze(-1)+1e-10 # Avoid division by zero
    dici["dR"] = K_tensor
        
    
    # Compute gradients
    grads, = torch.autograd.grad(
        outputs=output, 
        inputs=input, 
        grad_outputs=torch.ones_like(output, device=device), 
        create_graph=train
    )
     
    with contextlib.nullcontext() if train else torch.no_grad():
        # Build the variation tensor [B,6]
        init_key = "logE"
        var_tensor = torch.einsum('b n, b n k -> b k', grads[:,:,dict_vars[init_key]], dici[init_key])
        
        for key in dict_vars.keys():
            if key!=init_key:
                var_tensor += torch.einsum('b n, b n k -> b k', grads[:,:,dict_vars[key]], dici[key])
#             var_tensor = torch.einsum('b n, b n k -> b k',grads[:,:,0], (trans_dict['eta'] - trans_dict_jet['eta']))
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,1], (trans_dict['phi'] - trans_dict_jet['phi']))
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,2], (trans_dict['logPt'] - trans_dict_jet['logPt']))
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,3], trans_dict['logPt'])
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,4], (trans_dict['logE'] - trans_dict_jet['logE']))
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,5], trans_dict['logE'])

#             # delta R calculation (K_tensor)
#             K_tensor = (input[:,:,0].unsqueeze(-1) * (trans_dict['eta'] - trans_dict_jet['eta']))
#             K_tensor += (input[:,:,1].unsqueeze(-1) * (trans_dict['phi'] - trans_dict_jet['phi']))
#             K_tensor /= input[:,:,6].unsqueeze(-1)+1e-10 # Avoid division by zero
#             var_tensor += torch.einsum('b n, b n k -> b k',grads[:,:,6], K_tensor)


        # Apply generators mask if provided
        if generators is not None:
            generators_tensor = torch.tensor(generators, dtype=torch.bool, device=device).unsqueeze(0)  # [1, 6]
            var_tensor = torch.where(generators_tensor, var_tensor, torch.zeros_like(var_tensor))

        
        # Compute the loss
        loss = torch.norm(var_tensor, p=2, dim=1)**2
        
        if take_mean:
            loss = loss.mean()
        return loss



class SymmLoss_pT_eta_phi(nn.Module):

    def __init__(self,model, gens_list = ["Lx", "Ly", "Lz", "Kx", "Ky", "Kz"],device="cpu"):
        super(SymmLoss_pT_eta_phi, self).__init__()
        
        self.device = device
        self.model = model.to(device)
        
        # Initialize generators (in future add different reps for inputs?)
        GenList_names = []
        Lorentz_names = ["Lx", "Ly", "Lz", "Kx", "Ky", "Kz"]
        for gen in gens_list:
            if gen in Lorentz_names:
                GenList_names.append(gen)
            else:
                print(f"generator \n {gen} needs to be one of: {Lorentz_names}") #This is for now. Later will add a part that deals with calculating the transforamtion for a given generator. 
                
                # self.generators = einops.rearrange(gens_list, 'n w h -> n w h')
                # self.generators = self.generators.to(device)
        self.generators = GenList_names
        
        

    def forward(self, input, model_rep='scalar',norm = "none",nfeatures = "", mask=None, log_E=False, log_pT=False):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)

        # input_p4 = input
        if nfeatures!="":
            dim = nfeatures
        else:
            dim = 4 #self.generators.shape[-1]
        #Assuming input is shape [B,d*N] d is the number of features, N is the number of particles
        # input_reshaped = einops.rearrange(input, '... (N d) -> ... N d',d = dim)
        
        E =  torch.exp(input[:,:,0]) if log_E else input[:,:,0]  #assuming input features are ordered as (E,pT,eta,phi)
        
        pT = torch.exp(input[:,:,1]) if log_pT else input[:,:,1]
        
        eta = input[:,:,2]
        
        phi = input[:,:,3]
        
        GenList = self.generators

        # Add back in masked particles
        E = torch.masked_fill(E, ~mask, 0)
        pT = torch.masked_fill(pT, ~mask, 0)

        
        #dvar/dp L p, 
        ngen = len(self.generators)
        dE = torch.zeros_like(E).to(self.device)
        dpT = torch.zeros_like(pT).to(self.device)
        deta = torch.zeros_like(eta).to(self.device)
        dphi = torch.zeros_like(phi).to(self.device)
        
        
        #Here for all the Lorentz generators. Later can add options for only some of them.
        dE   = {"Lx": torch.zeros_like(E),              "Ly": torch.zeros_like(E),                "Lz":  torch.zeros_like(E),  "Kx":pT*torch.cos(phi),                    "Ky":pT*torch.sin(phi),                    "Kz":pT*torch.sinh(eta)}
        dpT  = {"Lx": pT*torch.sin(phi)*torch.sinh(eta),"Ly": -pT*torch.cos(phi)*torch.sinh(eta), "Lz":  torch.zeros_like(pT), "Kx":E*torch.cos(phi),                     "Ky":E*torch.sin(phi),                     "Kz":torch.zeros_like(pT)}
        deta = {"Lx": -1*torch.sin(phi)*torch.cosh(eta),  "Ly": torch.cos(phi)*torch.cosh(eta),     "Lz":  torch.zeros_like(eta),"Kx":-E*torch.cos(phi)*torch.tanh(eta)/pT, "Ky":-E*torch.sin(phi)*torch.tanh(eta)/pT, "Kz":E/(pT*torch.cosh(eta))}
        dphi = {"Lx":  torch.cos(phi)*torch.sinh(eta),  "Ly": torch.sin(phi)*torch.sinh(eta),     "Lz":-1*torch.ones_like(phi),"Kx":-E*torch.sin(phi)/pT,                 "Ky":E*torch.cos(phi)/pT,                  "Kz":torch.zeros_like(phi)}

        
        varsE = torch.empty(ngen,E.shape[0],E.shape[1]).to(self.device)
        varspT = torch.empty(ngen,E.shape[0],E.shape[1]).to(self.device)
        varseta = torch.empty(ngen,E.shape[0],E.shape[1]).to(self.device)
        varsphi = torch.empty(ngen,E.shape[0],E.shape[1]).to(self.device)
            
        for i,gen in enumerate(GenList):
            varsE[i] = dE[GenList[i]] /E if log_E else dE[GenList[i]]
            varspT[i] = dpT[GenList[i]] /pT if log_pT else dpT[GenList[i]]
            varseta[i] = deta[GenList[i]]
            varsphi[i] = dphi[GenList[i]]
        
        varsSymm = torch.stack((varsE,varspT,varseta,varsphi), dim = -1) #[n,B,N,d]

        # Deal with Nans from expressions like 0/0
        varsSymm = torch.nan_to_num(varsSymm, posinf=0., neginf=0.)
        
        #print(varsSymm.shape)
            
        # Compute model output, shape [B]
        output = self.model(input, mask)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads_input, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)

        
        
        # Reshape grads to [B, N, d]
        grads_input = grads_input.unsqueeze(0)
       
        
        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', varsSymm, grads_input[:,:,:,:4]) # Input has 9 dimensions, only care about first 4 (LogE, LogpT, eta, phi)
        
        scalar_loss = (differential_trans ** 2).mean()
        
        return scalar_loss


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_size):
        super().__init__()

        self.embed = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
        )

        
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        # self.encoder_layer_4 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        
        
        self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
        )
    
    def forward(self, x, mask=None):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x = self.embed(x)
            x = self.encoder_layer_1(x)
            x = self.encoder_layer_2(x)
            x = self.encoder_layer_3(x)
            # x = self.encoder_layer_4(x)
            x = torch.sum(x*mask.unsqueeze(-1), axis=1)
            return self.classifier(x)
        
        
        
class DeepSet(nn.Module):
    def __init__(self, input_dim, rho_size, phi_size,):
        super().__init__()

        self.rho = nn.Sequential(
                nn.Linear(input_dim, rho_size),
                nn.ReLU(),
                nn.Linear(rho_size, rho_size),
                nn.ReLU(),
                nn.Linear(rho_size, rho_size),
                nn.ReLU(),
                nn.Linear(rho_size, rho_size)
        )

        self.phi = nn.Sequential(
                nn.Linear(rho_size, phi_size),
                nn.ReLU(),
                nn.Linear(phi_size, phi_size),
                nn.ReLU(),
                nn.Linear(phi_size, phi_size),
                nn.ReLU(),
                nn.Linear(phi_size, 1),
                nn.Sigmoid(),
        )
    def forward(self, x, mask=None):
        x = self.rho(x)
        x = torch.sum(x*mask.unsqueeze(-1), axis=1)
        return self.phi(x)
    

class JetDataset(Dataset):
    def __init__(self, X, y, mask, X_cylindrical=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = mask
        self.X_cylindrical = X_cylindrical
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.X_cylindrical != None:
            return self.X[idx], self.y[idx], self.mask[idx], self.X_cylindrical[idx]
        else:
            return self.X[idx], self.y[idx], self.mask[idx]
        

def get_jet_relvars(four_vec, four_vec_cy):
    
    jet_four_vec = torch.sum(four_vec, dim=1)
    jet_four_vec_cy = to_cylindrical(jet_four_vec.unsqueeze(1))

    # log(E_const/E_jet)
    log_Er = four_vec_cy[:,:,0] - jet_four_vec_cy[:,:,0]

    # log(pt_const/pt_jet)
    log_ptr = four_vec_cy[:,:,1] - jet_four_vec_cy[:,:,1]

    # dEta
    dEta = jet_four_vec_cy[:,:,2] - four_vec_cy[:,:,2] 

    # dPhi
    dPhi = jet_four_vec_cy[:,:,3] - four_vec_cy[:,:,3] 

    # dR
    dR = torch.sqrt(dEta**2 + dPhi**2)

    jet_features = torch.cat([log_Er.unsqueeze(-1), log_ptr.unsqueeze(-1), dEta.unsqueeze(-1), dPhi.unsqueeze(-1), dR.unsqueeze(-1)], axis=2)

    zero_mask = (four_vec == 0.0).any(dim=-1, keepdim=True)
    zero_mask = zero_mask.expand_as(jet_features)
    jet_features[zero_mask] = 0.0
    
    return jet_features
    

def boost_3d(data, device="cpu", beta=None, beta_max=0.95):

    # sample beta from sphere
    b1 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    b2 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    theta = 2 * np.pi * b1
    phi = np.arccos(1 - 2 * b2)
    
    beta_x = np.sin(phi) * np.cos(theta)
    beta_y = np.sin(phi) * np.sin(theta)
    beta_z = np.cos(phi)
    
    beta = torch.cat([beta_x.unsqueeze(-1),beta_y.unsqueeze(-1), beta_z.unsqueeze(-1)], axis=1)
    bf = torch.tensor(np.random.uniform(0, beta_max, size=(len(data),1)), dtype=torch.float32)
    bf = bf#**(1/2)
    beta = beta*bf#beta*beta_max
    
    beta_norm = torch.norm(beta, dim=1) 

    # make sure we arent violating speed of light
    #assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
    L = torch.zeros((len(data), 4, 4)).to(device)
    L[:,0, 0] = gamma
    L[:,1:, 0] = L[:,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[:, 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('bi,bj->bij', (beta, beta))/ beta_squared[...,None, None]
    
    assert torch.all (torch.linalg.det(L)) == True

    boosted_four_vector = torch.einsum('bij,bkj->bik', L.type(torch.float32), data.type(torch.float32)).permute(0, 2, 1) 

    # Validate that energy values remain non-negative
    assert torch.all(boosted_four_vector[:, :, 0] >= 0), "Negative energy values detected!"
    
    return boosted_four_vector

def boost(data, pdgid=None, device="cpu", beta=None):
    # print(device)
    if beta is None:
        beta = torch.tensor(np.random.uniform(0, 0.99, size=len(data)), dtype=torch.float32)
    else:
        beta = beta*torch.ones(len(data))
        
    beta = beta.repeat(data.shape[1])
    gamma = (1-beta*beta)**(-0.5)

    beta = beta.to(device)
    gamma = gamma.to(device)

    E_b = gamma*(data[:,:,0].flatten()- beta* data[:,:,1].flatten() )
    px_b = gamma*(data[:,:,1].flatten() - beta* data[:,:,0].flatten())
    
    E_b = E_b.reshape(data.shape[0], data.shape[1])
    px_b = px_b.reshape(data.shape[0], data.shape[1])

    four_vec = torch.cat([ E_b.unsqueeze(-1), px_b.unsqueeze(-1), data[:,:,2:4]], axis = 2)
    four_vec = four_vec.to(device)

    # return  torch.cat([four_vec, data[:,:,4:]], axis = 2)
    return four_vec
        
    
    
    
    
class genNet(nn.Module):
    def __init__(self, input_size=4, output_size=1, hidden_size=10, n_hidden_layers=3, init="rand", equiv="False", rand = "True", freeze = "False",activation = "ReLU", skip = "False", seed = 98235):
        super().__init__()
        self.equiv=equiv
        self.skip = skip
        if rand=="False":
            np.random.seed(seed)
            torch.manual_seed(seed)

        if activation =="sigmoid":
            #input layer
            module_list = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.Sigmoid()])
        elif activation== "GeLU":
            module_list = [nn.Linear(input_size, hidden_size), nn.GELU()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])    

        else:
            #input layer
            module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

        dinput = input_size
        if self.equiv=="True":
            bi_tensor = torch.randn(input_size,input_size)

            if init=="eta":
                diag = torch.ones(dinput)*(-1.00)
                diag[0]=1.00
                bi_tensor = torch.diag(diag)

            elif init=="delta":
                bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
            bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
            self.bi_tensor = bi_tensor.to(devicef)
            
            if (freeze =="False" or freeze == False):
                
                self.bi_tensor = torch.nn.Parameter(bi_tensor.to(devicef))
                self.bi_tensor.requires_grad_()

            self.equiv_layer = nn.Linear(1, input_size)
            self.skip_layer = nn.Linear(input_size, input_size)
            

    def forward(self,x):

        if self.equiv=="True":
            y = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x).unsqueeze(1)
            
            y = self.equiv_layer(y)
            
            if self.skip =="True":
                y = y + self.skip_layer(x)
        else:
            y = x

        return self.sequential(y)
    

class top_data():
            
    def get_dataset(self,data_file = "train.h5",batch_size=512,shuffle=True, get_train_data = True, get_test_data = True, set_seed = True, seed = 0,nj = 500000):
        
        self.seed = seed
        self.data_file = data_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if set_seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # load data
        self.batch_size = batch_size
        data =  h5.File(data_file, 'r')
        
        X = data["table"]["table"]["values_block_0"]
        Y = data["table"]["table"]["values_block_1"]
        Y = Y[:,1]
        
        X_QCD = X[np.where(Y==0)]
        Y_QCD = Y[np.where(Y==0)]
        X_top = X[np.where(Y==1)]
        Y_top = Y[np.where(Y==1)]
        
        X_QCD_const = X_QCD[:,:800].reshape(len(X_QCD), 200, 4)
        X_top_const = X_top[:,:800].reshape(len(X_top), 200, 4)
        X_top_truth = X_top[:,800:].reshape(len(X_top), 4)
        
        mask_QCD = np.all(X_QCD_const != 0, axis=2)
        mask_top = np.all(X_top_const != 0, axis=2)
        
        #nj = 500_000
        X = np.concatenate([X_QCD_const, X_top_const])
        Y = np.concatenate([Y_QCD, Y_top])
        mask = np.concatenate([mask_QCD, mask_top])
        
        X, Y, mask = shuffle_fun(X, Y, mask)
        
        if get_train_data:
            train_X = X[:nj]
            train_Y = Y[:nj]
            train_mask = mask[:nj]

            train_X_cylindrical = to_cylindrical(torch.tensor(train_X, dtype=torch.float32))

            train_jet_vars = get_jet_relvars(torch.tensor(train_X, dtype=torch.float32), train_X_cylindrical)

            train_X_cylindrical = torch.cat([train_X_cylindrical,train_jet_vars], axis=2)

            data = JetDataset(X=train_X, y=train_Y, mask=train_mask, X_cylindrical=train_X_cylindrical)

            self.train_data = data

            self.train_loader = DataLoader(data,batch_size=self.batch_size,shuffle=shuffle)
        
        if get_test_data:
            test_X = X[nj:2*nj]
            test_Y = Y[nj:2*nj]
            test_mask = mask[nj:2*nj]
            test_X_cy = to_cylindrical(torch.tensor(test_X, dtype=torch.float32))
            test_jet_vars = get_jet_relvars(torch.tensor(test_X, dtype=torch.float32), test_X_cy)
            test_X_cy = torch.cat([test_X_cy,test_jet_vars], axis=2)

            test_data = JetDataset(test_X,test_Y, test_mask, test_X_cy)
            self.test_data = test_data

            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle)
        
        
        

class top_symm_net_train():

    def __init__(self,gens_list=gens_Lorentz,input_size = 4,init = "rand",equiv="False",rand="False",freeze = "False", activation = "ReLU", skip="False",broken_symm = "False",spurions = torch.diag(torch.tensor([0.00,0.00,0.00,0.00]))):
        self.train_loss_vec = []
        self.symm_loss_vec = []
        self.tot_loss_vec = []
        self.running_loss = 0.0
        self.symm_loss = 0.0
        self.train_loss_lam = {}
        self.symm_loss_lam = {}
        self.tot_loss_lam = {}
        self.models = {}
        self.models_best_tot = {}
        self.models_best_symm = {}
        self.models_best_MSE = {}

        self.init = init
        self.equiv = equiv
        self.rand=rand
        self.freeze = freeze, 
        self.activation = activation
        self.skip=skip
        self.input_size = input_size
        self.gens_list=gens_list
        
        self.dataset_seed = 0
        self.train_seed = 0
        self.N = 0
        
        self.broken_symm = broken_symm
        self.spurions = spurions

    
        
    def prepare_dataset(self,data_file = "train.h5",batch_size=512,shuffle=True,set_seed = True, seed = int(torch.round(torch.rand(1)*10000)), nj = 500000):
        
        self.data_seed = seed
        self.data_file = data_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nj = nj
        
        
        # load data
        
        data = top_data()
        data.get_dataset(data_file = self.data_file,batch_size=self.batch_size,shuffle=self.shuffle, get_train_data = True, get_test_data = False, set_seed = True, seed = self.data_seed, nj = self.nj)
        
        return data.train_loader
        
   
        
        

    def set_model(self,gens_list=gens_Lorentz, ML_model = "DeepSet",input_dim = 9, rho_size = 256, phi_size = 128, input_size=4, embed_dim=256, hidden_size=128,rand="False",activation = "ReLU"):
       
        
        self.rand=rand
        
        self.activation = activation
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.n_hidden_layers = n_hidden_layers
        
        self.ML_model = ML_model
        
        if self.ML_model == "DeepSet":
            self.input_dim = input_dim
            self.rho_size = rho_size
            self.phi_size = phi_size
            
        if self.ML_model == "Transformer":
            self.input_size = input_size
            self.embed_dim = embed_dim
            self.hidden_size = hidden_size
            
        
    def train_model(self,model, dataloader, criterion, penalty,optimizer, nepochs=15, device='cpu', apply_symm=False,lambda_symm = 1.0, apply_MSE = False):

        model.to(device)
        #symmLoss = SymmLoss_pT_eta_phi(model, device=device)
        num_epochs = nepochs
        # save the losses
        apply_symm = (apply_symm==True) or (apply_symm=="True")
        apply_MSE = (apply_MSE==True) or (apply_MSE=="True")
        
        self.apply_MSE = apply_MSE
        self.apply_symm = apply_symm
        
        loss_tracker = {
            "Loss": [],
            "BCE": [],
            "Symm_Loss": [],
            "beta": [],
            "MSE_Loss": []
        }
        failed_jets = []
        print(lambda_symm)
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            rbce = 0.0
            rsymm = 0.0
            rmse = 0.0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}")):
                optimizer.zero_grad()  # Zero the gradients

                batch = [x.to(device) for x in batch]
                
                X, y, mask, X_cy = batch
                
                # X = X.to(device)
                # y = y.to(device)
                # X_cy = X_cy.to(device)
                # mask = mask.to(device)                

                outputs = model(X_cy, mask)
                bce = criterion(outputs.squeeze(), y)
                
                X_boost = boost_3d(X, device)
                X_boost_cy = to_cylindrical(X_boost)
                boost_jet_vars = get_jet_relvars(X_boost, X_boost_cy)
                X_boost_cy = torch.cat([X_boost_cy, boost_jet_vars], axis=2)

                optimizer.zero_grad()  # Zero the gradients

                outputs_boost = model(X_boost_cy, mask)

                # catch NaNs
                output_nan = torch.sum(torch.isnan(outputs_boost))

                if output_nan > 0:
                    print(f"Nan found in output in batch: {batch_idx}, Nans: {output_nan}")
                    outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)

                mse = penalty(outputs.squeeze(), outputs_boost.squeeze())
                            
                symm = symm_loss_scalar9(model, X_cy, X_cartesian = X, mask=mask,train = apply_symm,take_mean = True)#symmLoss(X_cy, mask=mask)

                # print(symm)

                loss =  bce
                if apply_symm:
                    loss += lambda_symm*symm 
                if apply_MSE:
                    loss += mse 
                
                if torch.isinf(symm):
                    print("Found infinity")

                loss.backward()  # Backward pass
                # gradients too large? 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

                optimizer.step()  # Update parameters

                running_loss += loss.item()
                rbce += bce.item()
                rsymm += symm.item()
                rmse += mse.item() 
                # rbeta += beta.item()
                # break


            running_loss /=len(dataloader)
            rbce /= len(dataloader)
            rsymm /= len(dataloader)
            # rbeta /= len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, BCE: {rbce:.4f}, Symm: {rsymm}')
            loss_tracker["Loss"].append(running_loss)
            loss_tracker["BCE"].append(rbce)
            loss_tracker["Symm_Loss"].append(rsymm)
            loss_tracker["MSE_Loss"].append(rmse)
            # loss_tracker["beta"].append(rbeta)
            # break
            model_clone = copy.deepcopy(model)


        return loss_tracker,model_clone
    
    
    def run_training(self,lam_vec, dataloader, criterion = torch.nn.BCELoss(), penalty = torch.nn.MSELoss(),opt = "Adam",lr = 5e-4, nepochs=15, device='cpu', apply_symm=False, apply_MSE=False,set_seed = True,seed = int(torch.round(torch.rand(1)*10000))):
        
        self.lr = lr
        self.opt = opt
        self.apply_symm = apply_symm
        self.apply_MSE = apply_MSE
        self.train_seed = seed
        
        if set_seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_loader_copy = copy.deepcopy(dataloader)
        if self.ML_model == "DeepSet":
            model = DeepSet(input_dim=self.input_dim, rho_size= self.rho_size, phi_size=self.phi_size)
        if self.ML_model == "Transformer":
            model = Transformer(input_dim = self.input_dim, embed_dim = self.embed_dim, hidden_size = self.hidden_size)
        train_outputs = {}
        models = {}

        for lam_val in lam_vec:
            train_loader_copy = copy.deepcopy(dataloader)
            if set_seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            model_train = copy.deepcopy(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total model parameters: {total_params}")
            
            # Define the loss function and optimizer
            if self.opt == "SGD":
                optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum = 0.9)

            elif self.opt =="GD":
                optimizer = optim.GD(model_train.parameters(), lr=lr)
            
            else:
                optimizer = optim.Adam(model_train.parameters(), lr=lr)


            train_output,model_trained = self.train_model(model = model_train, dataloader = train_loader_copy, criterion = criterion, penalty = penalty ,optimizer = optimizer, device="cuda",apply_symm=apply_symm,apply_MSE=apply_MSE,lambda_symm = lam_val,nepochs=nepochs)

            models[lam_val] = copy.deepcopy(model_trained)
            train_outputs[lam_val] = copy.deepcopy(train_output)
            
        self.models = models
        self.train_outputs = train_outputs

        return train_outputs,models
                
                    
                    


class top_analysis_trained(top_symm_net_train):

    def __init__(self):
        super().__init__()
        
    def get_trained(self, trained_net):
        self.__dict__ = {key:copy.deepcopy(value) for key, value in trained_net.__dict__.items()}
        
    
    def title(self):
        text = f"top_{self.ML_model}_symm_{self.apply_symm}_MSE_{self.apply_MSE}"# hidden size:{self.hidden_size} layers:{self.n_hidden_layers} activation:{self.activation} lr:{self.lr} opt:{self.opt} "
        # self.spurions_for_print = ""
        # if self.broken_symm == "True" or self.broken_symm == True:
        #     text=f"{text} broken symm"
        #     spurions_for_print = "spurions:\n"
        #     if self.input_spurions == "True" or self.input_spurions == True:
        #         text = f"{text} input spurions"
        #     for spurion in self.spurions:
        #         spurions_for_print+= f"{spurion}\n"
        #     self.spurions_for_print = spurions_for_print
        # if self.equiv== "True":
        #     text=f"{text} bi-linear layer"
        #     if self.skip =="True":
        #         text=f"{text} skip"
        #     if self.init=="eta" or self.init=="delta":
        #         text=f"{text} init: {self.init}"
        #     if self.freeze=="True":
        #         text=f"{text} freeze"
        # if self.symm_norm == "True" or self.symm_norm == True:
        #         text = f"{text} norm"
        self.title_text = text
        self.filename = self.title_text.replace(" ","_").replace(":","_")+f"train_seed_{self.train_seed}"+f"_data_seed_{self.data_seed}"
        return text
    
    def save_trained(self,outdir = "./storage"):
        filename = self.title_text.replace(" ","_").replace(":","_")+f"train_seed_{self.train_seed}"+f"_data_seed_{self.data_seed}"
        with open(f'{outdir}/{filename}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self, f)

    def plot_losses(self,save = False, outdir = "./",filename = "",print_spurions = False):

        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_outputs
        #symm_loss_lam = self.symm_loss_lam
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            losses = self.train_outputs[lam_val]
            plt.semilogy(range(len(losses["BCE"])),losses["BCE"],label = rf"$\lambda$ = {lam_val}, BCE", color = color_vec[i%len(color_vec)])
            plt.semilogy(range(len(losses["Symm_Loss"])),losses["Symm_Loss"],label = rf"$\lambda$ = {lam_val}, symm", color = color_vec[i%len(color_vec)],ls = "--")
            if "MSE_Loss" in losses.keys():
                plt.semilogy(range(len(losses["MSE_Loss"])),losses["MSE_Loss"],label = f"MSE", color = color_vec[i%len(color_vec)],ls = "-.")
                
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_losses_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")
        
    def plot_losses_side(self,save = False, outdir = "./",filename = ""):
        nfigs = 4 if "MSE_Loss" in self.train_outputs[0.0].keys() else 3
        
        fig, ax = plt.subplots(1,nfigs, figsize=(12,4))
        for lam_val in self.train_outputs.keys():
            losses = self.train_outputs[lam_val]
            # Total Loss
            lam = f"{lam_val:.1e}"
            ax[0].plot(losses["Loss"], label=rf"$\lambda = {lam}$")
            #ax[0].plot(losses_symm["Loss"], label="Symm Loss Applied")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Total Loss")
            ax[0].legend()

            # BCE Component
            ax[1].plot(losses["BCE"])
            #ax[1].plot(losses_symm["BCE"])
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("BCE Loss")

            # symm Component
            ax[2].semilogy(losses["Symm_Loss"])
            #ax[2].plot(losses_symm["Symm_Loss"])
            ax[2].set_xlabel("Epoch")
            ax[2].set_ylabel("Symm Loss")
            
            # MSE Component
            if "MSE_Loss" in losses.keys():
                ax[3].plot(losses["MSE_Loss"])
                #ax[3].plot(losses_symm["Symm_Loss"])
                ax[3].set_xlabel("Epoch")
                ax[3].set_ylabel("MSE Loss")

        fig.tight_layout()
        if save==True or save=="True":
            if filename =="":
                filename = "top_tagging_loss_100k_symmLoss_applied_"+self.filename
        plt.savefig(f"{outdir}/{filename}.pdf")
            
    
    def plot_symm_loss(self,save = False, outdir = "./",filename = "",print_spurions = False):
        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        #train_loss_lam = self.train_loss_lam
        symm_loss_lam = self.symm_loss_lam
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            #plt.semilogy(range(len(train_loss_lam[lam_val])),train_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, MSE", color = color_vec[i])
            plt.semilogy(range(len(symm_loss_lam[lam_val])),symm_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, symm", color = color_vec[i%len(color_vec)])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("symm loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_symm_losses_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")

    
    def plot_task_loss(self,save = False, outdir = "./",filename = "",print_spurions = False):
        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_outputs
        #symm_loss_lam = self.symm_loss_lam
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            plt.semilogy(range(len(train_loss_lam[lam_val])),train_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, MSE", color = color_vec[i%len(color_vec)])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("MSE loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_MSE_losses_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")

            
    def evaluate_model(self,model, dataloader, device='cpu', beta_max = [1.0]):
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        
        pred = []
        true = []
        boost_ = {mb: [] for mb in beta_max} 
        pt = []
        pt_boost = {mb: [] for mb in beta_max}
        with torch.no_grad():  # Disable gradient calculation
            for idx, batch in enumerate(dataloader):
                X, y, mask, X_cy = batch

                X = X.to(device)
                y = y.to(device)
                mask = mask.to(device)
                X_cy = X_cy.to(device)
                outputs = model(X_cy, mask)  # Forward pass
                pred.append(outputs)
                true.append(y)
                jet_cy = to_cylindrical(torch.sum(X, dim=1).unsqueeze(1))
                pt.append(jet_cy[:,:,1])

                for mb in beta_max:
                    if idx ==0:
                        print(f"Max beta: {np.sqrt(mb)}")
                    X_boost = boost_3d(X, device=device, beta_max=mb)
                    jet_cy = to_cylindrical(torch.sum(X_boost, dim=1).unsqueeze(1))
                    X_boost_cy = to_cylindrical(X_boost)
                    jet_vars = get_jet_relvars(X_boost, X_boost_cy)
                    X_boost_cy = torch.cat([X_boost_cy,jet_vars], axis=2)                         
                    
                    outputs_boost = model(X_boost_cy, mask)
                    boost_[mb].append(outputs_boost)
                    pt_boost[mb].append(jet_cy[:,:,1])

        model.to("cpu")
        for mb in beta_max:
            boost_[mb] = torch.cat(boost_[mb])

        return torch.cat(pred), boost_, torch.cat(true), torch.cat(pt), pt_boost
    
    
    
    def get_metrics(self,pred, true, boost, beta_max):
        bm = {"auc": [], "acc": [], "boost_auc": [], "boost_acc":[] }

        fpr, tpr, threshold = roc_curve(true.cpu(), pred.cpu())
        roc_auc = bm["auc"].append(auc(fpr, tpr))
        acc = accuracy_score(true.cpu(), (pred >= 0.5).int().cpu())
        bm["acc"].append(acc)

        for mb in beta_max:
            fprb, tprb, _ = roc_curve(true.cpu(), boost[mb].cpu())
            roc_aucb = auc(fprb, tprb)
            accb = accuracy_score(true.cpu(), (boost[mb] >= 0.5).int().cpu())
            bm["boost_auc"].append(roc_aucb)
            bm["boost_acc"].append(accb)
        return bm
    
    
    
    def pred_plot(self,save = False, outdir = "./",filename = "",print_spurions = False):
        inputs = self.train_data.to(devicef)
        plt.clf()
        fig = {}
        for lam_val in self.models.keys():
            plt.clf()
            fig[lam_val] = plt.figure()
            plt.scatter(self.train_labels.cpu().squeeze(),self.models[lam_val](inputs).detach().cpu().squeeze(),label = rf"$\lambda$ = {lam_val}")

            plt.legend()
            plt.xlabel("truth")
            plt.ylabel("pred")
            
            if print_spurions == "True" or print_spurions == True:
                plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
            
            if save==True or save=="True":
                if filename =="":
                    file = f"plot_pred_lam_{lam_val}_{self.filename}"
                else:
                    file = filename
                fig[lam_val].show()
                fig[lam_val].savefig(f"{outdir}/{file}.pdf")
            #plt.show()
                plt.close(fig[lam_val])
                
    
        
    def evaluate_models(self,test_loader):
        
        pred = {}
        boost = {}
        true = {}
        pt = {}
        boost_pt = {}
        models = self.models
        
        seed = self.data_seed
        
        for lam_val in models.keys():
            np.random.seed(seed)
            torch.manual_seed(seed)
            pred[lam_val], boost[lam_val], true[lam_val], pt[lam_val], boost_pt[lam_val] = self.evaluate_model(model = models[lam_val], dataloader = copy.deepcopy(test_loader), device="cuda")
            
        self.pred = pred
        self.boost = boost
        self.true = true
        self.pt = pt
        self.pt_boost = boost_pt
        
    def print_AUC(self,beta_max = 0.8):
        
        for lam_val in models.keys():
            bm = self.get_metrics(self.pred[lam_val], self.true[lam_val], self.boost[lam_val], beta_max)
            print(rf"$\lambda$ = {lam_val}: AUC =  {bm['auc'][0]},boosted AUC =  {bm['boost_auc'][0]}" )
        
        
    def plot_hists(self,save = False, outdir = "./",filename = ""):
        fig, ax = plt.subplots(1,3, figsize=(12,4))

        for lam_val in pred.keys():
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            h1, xedges, yedges, img1= ax[0].hist2d(pred[0.0].cpu().numpy().flatten(), boost[0.0][0.64].cpu().numpy().flatten(), bins=50, norm=LogNorm())
            h2, _, _, img2  = ax[1].hist2d(pred[lam_val].cpu().numpy().flatten(), boost[lam_val][0.64].cpu().numpy().flatten(), bins=50, norm=LogNorm());

            hdiff = h2 - h1

            img3= ax[2].imshow(hdiff.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],  aspect='auto', norm=LogNorm())
            ax[0].set_ylabel("$p_{Boost}$", fontsize=15)
            ax[0].set_xlabel("$p_{No Boost}$", fontsize=15)
            ax[1].set_xlabel("$p_{No Boost}$", fontsize=15)
            ax[2].set_xlabel("$p_{No Boost}$", fontsize=15)

            ax[0].set_title("Baseline", fontsize=15)
            ax[1].set_title("Symm Loss", fontsize=15)
            ax[2].set_title("Hist Difference", fontsize=15)

            fig.colorbar(img1, ax=ax[0], orientation='vertical')
            fig.colorbar(img2, ax=ax[1], orientation='vertical')
            fig.colorbar(img3, ax=ax[2], orientation='vertical')
            plt.tight_layout()
            if save==True or save=="True":
                if filename =="":
                    file = f"plot_pred{ext}_lam_{lam_val}_{self.filename}"
                else:
                    file = filename
                fig[lam_val].show()
                fig[lam_val].savefig(f"{outdir}/hists_lam_{lam_val}_{file}_{self.filename}.pdf")
            plt.show()
                
    def pred_plot_ext(self,data,model = "last",save = False, outdir = "./",filename = "",print_spurions = False):
        inputs = self.train_data.to(devicef)
        plt.clf()
        fig = {}
        if model == "last":
            models = self.models
            ext = ""
        elif model== "symm":
            models = self.models_best_symm
            ext = "_best_symm"
        elif model == "MSE":
            models = self.models_best_MSE
            ext = "_best_MSE"
        elif model == "tot":
            models = self.models_best_tot
            ext = "_best_tot"
            
        if self.broken_symm == "True" or self.broken_symm == True:
            truth_new = Lorentz_myfun_broken(data,spurions = self.spurions)
        else:
            truth_new = Lorentz_myfun(data)
            
        for lam_val in self.models.keys():
            plt.clf()
            fig[lam_val] = plt.figure()
            plt.scatter(self.train_labels.cpu().squeeze(),models[lam_val](inputs).detach().cpu().squeeze(),label = rf"$\lambda$ = {lam_val} training data")
            plt.scatter(truth_new.cpu().squeeze(),models[lam_val](data).detach().cpu().squeeze(),label = rf"$\lambda$ = {lam_val} new data",color = "pink",alpha = 0.2)
            plt.plot(truth_new.cpu().squeeze(),truth_new.cpu().squeeze(),color = "black")
            plt.legend()
            plt.xlabel("truth")
            plt.ylabel("pred")
            err = ((truth_new.cpu().squeeze()-models[lam_val](data).detach().cpu().squeeze())**2).mean()
            #err = '%.4E' % Decimal("f{err}")
            err = "{:.4e}".format(err)
            mse = self.train_loss_lam[lam_val][-1]
            mse = "{:.4e}".format(mse)
            symm = self.symm_loss_lam[lam_val][-1]
            symm = "{:.4e}".format(symm)
            text = f"lam = {lam_val}, var = {err} MSE = {mse}, symm = {symm}"
            print(text)
            text = f"var = {err}, MSE = {mse}, symm = {symm}"
            plt.text(-9, -10,text)
            
            if print_spurions != "False" or print_spurions != False:
                plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
            
            if save==True or save=="True":
                if filename =="":
                    file = f"plot_pred{ext}_lam_{lam_val}_{self.filename}"
                else:
                    file = filename
                fig[lam_val].show()
                fig[lam_val].savefig(f"{outdir}/plot_pred_lam_{lam_val}_{file}_{self.filename}.pdf")
            plt.show()
                #plt.close(fig[lam_val])


        
            