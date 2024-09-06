import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

import numpy as np
import matplotlib.pyplot as plt
import copy
import dill as pickle
import os
import time

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


def Lorentz_myfun(input):
        m2 = torch.einsum("... i, ij, ...j -> ...",input, torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])), input)
        out = m2**2+15*m2
        return out.unsqueeze(1)

def Lorentz_myfun_broken(input,spurions = [torch.tensor([0.0,0.0,0.0,0.0])]):
    metric_tensor = torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00]))
    m2 = torch.einsum("... i, ij, ...j -> ...",input, metric_tensor, input)
    breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion, metric_tensor, input) for spurion in spurions]
    coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
    coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
    symm_out = m2**2+15*m2
    out = symm_out
    for i in range(len(breaking_scalars)):
        out += coeffs[i%len(coeffs)]*breaking_scalars[i]+coeffs_2[i%len(coeffs_2)]*breaking_scalars[i]**2
    return out.unsqueeze(1).to(devicef)




class SymmLoss(nn.Module):

    def __init__(self, gens_list,model,device = devicef):
        super(SymmLoss, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        # Initialize generators (in future add different reps for inputs?)
        self.generators = einops.rearrange(gens_list, 'n w h -> n w h')
        self.generators = self.generators.to(device)
        

    
    def forward(self, input, model_rep='scalar',norm = "none"):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)
        # Compute model output, shape [B]
        output = self.model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
        
        # Reshape grads to [B, N, d] 
        grads = einops.rearrange(grads, '... (N d) -> ... N d',d = self.generators.shape[-1])

        # Contract grads with generators, shape [n (generators), B, N, d]
        gen_grads = torch.einsum('n h d, ... N h->  n ... N d ',self.generators, grads)
        # Reshape to [n, B, (d N)]
        gen_grads = einops.rearrange(gen_grads, 'n ... N d -> n ... (N d)')

        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', gen_grads, input)

        
        # # Reshape grads to [B, N, d] 
        # grads = einops.rearrange(grads, '(N d) -> N d',d = self.generators.shape[-1])

        # # Contract grads with generators, shape [n (generators), B, N, d]
        # gen_grads = torch.einsum('n h d,  N h->  n N d ',self.generators, grads)
        # # Reshape to [n, B, (d N)]
        # gen_grads = einops.rearrange(gen_grads, 'n N d -> n (N d)')

        # # Dot with input [n ,B]
        # differential_trans = torch.einsum('n N, N -> n', gen_grads, input)

        scalar_loss = (differential_trans ** 2).mean()
     
            
        return scalar_loss
    
    


class SymmLoss_norm(nn.Module):

    def __init__(self, gens_list,model,device = devicef):
        super(SymmLoss_norm, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        # Initialize generators (in future add different reps for inputs?)
        self.generators = einops.rearrange(gens_list, 'n w h -> n w h')
        self.generators = self.generators.to(device)
        

    
    def forward(self, input, model_rep='scalar',norm = "none"):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)
        # Compute model output, shape [B]
        output = self.model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
        
        grads_norm = torch.einsum('... N, ... N -> ...', grads, grads)
        #print(grads_norm.mean())
        
        # Reshape grads to [B, N, d] 
        grads = einops.rearrange(grads, '... (N d) -> ... N d',d = self.generators.shape[-1])

        # Contract grads with generators, shape [n (generators), B, N, d]
        gen_grads = torch.einsum('n h d, ... N h->  n ... N d ',self.generators, grads)
        # Reshape to [n, B, (d N)]
        gen_grads = einops.rearrange(gen_grads, 'n ... N d -> n ... (N d)')

        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', gen_grads, input)
        
       
        
        # scalar_loss = (differential_trans ** 2).mean()
        # print(f"symm loss = {scalar_loss}")
        
        scalar_loss_norm = (1/len(self.generators))*(torch.sum(differential_trans**2,dim = 0)/grads_norm).mean()
     
            
        return scalar_loss_norm




class inv_model(nn.Module):

    def __init__(self,dinput = 4, doutput = 1,init = "rand"):
        super(inv_model,self).__init__()

        bi_tensor = torch.randn(dinput,dinput)

        if init=="eta":
            diag = torch.ones(dinput)*(-1.00)
            diag[0]=1.00
            bi_tensor = torch.diag(diag)

        elif init=="delta":
            bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
        
        bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
        self.bi_tensor = torch.nn.Parameter(bi_tensor)
        self.bi_tensor.requires_grad_()

    def forward(self,x, sig = "euc", d = 3 ):
        #y = x @ (self.bi_tensor @ x.T)
        y = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x)
        # x = torch.dot(x,y)
        return y

    
    
class broken_model(nn.Module):
    
    def __init__(self,dinput = 4, doutput = 1,init = "rand",spurions = [torch.tensor([0.0,0.0,0.0,0.0])]):
        super(broken_model,self).__init__()

        bi_tensor = torch.randn(dinput,dinput)

        if init=="eta":
            diag = torch.ones(dinput)*(-1.00)
            diag[0]=1.00
            bi_tensor = torch.diag(diag)

        elif init=="delta":
            bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
        
        bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
        self.bi_tensor = torch.nn.Parameter(bi_tensor)
        self.bi_tensor.requires_grad_()
        self.spurions = spurions
        self.coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
        self.coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
        
    def forward(self,x, sig = "euc", d = 3 ):
        #y = x @ (self.bi_tensor @ x.T)
        m2 = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x)
        breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion.to(devicef), self.bi_tensor, x) for spurion in self.spurions]
        
        symm_out = m2**2+15*m2
        out = symm_out
        for i in range(len(breaking_scalars)):
            out += self.coeffs[i%len(self.coeffs)]*breaking_scalars[i]+self.coeffs_2[i%len(self.coeffs_2)]*breaking_scalars[i]**2
            # x = torch.dot(x,y)
        return out


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
        
        

class symm_net_train():

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

#     def Lorentz_myfun(input):
#         m2 = torch.einsum("... i, ij, ...j -> ...",input, torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])), input)
#         out = m2**2+15*m2
#         return out.unsqueeze(1)
    
#     def Lorentz_myfun_broken(input,spurions = [torch.tensor([0.0,0.0,0.0,0.0])]):
#         metric_tensor = torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])).to(devicef)
#         m2 = torch.einsum("... i, ij, ...j -> ...",input, metric_tensor, input)
#         breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion.to(devicef), metric_tensor, input) for spurion in spurions]
#         coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
#         coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
#         symm_out = m2**2+15*m2
#         out = symm_out
#         for i in range(len(breaking_scalars)):
#             out += coeffs[i%len(coeffs)]*breaking_scalars[i]+coeffs_2[i%len(coeffs_2)]*breaking_scalars[i]**2
#         return out.unsqueeze(1).to(devicef)

    def prepare_dataset(self, N = 1000, dinput = 4, norm = 1, true_func = Lorentz_myfun, batch_size="all", shuffle=False, seed = 98235, broken_symm = "False", spurions = [[0.0,0.0,0.0,0.0]],input_spurions = "False"):
        
        self.N = N
        self.dataset_seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        train_data = (torch.rand(N,dinput)-0.5)*norm
        
        self.broken_symm = broken_symm
        self.spurions = [torch.tensor(spurion) for spurion in spurions]
        self.spurions_list = spurions
        self.true_func = lambda input: true_func(input,spurions = self.spurions) if (broken_symm == "True" or broken_symm == True) else true_func
        self.input_spurions = input_spurions
        
        
        train_labels = self.true_func(train_data).squeeze() #if (broken_symm == "True" or broken_symm == True) else true_func(train_data).squeeze()
        
        if batch_size=="all":
            batch_size = N

        if self.input_spurions == "True" or self.input_spurions==True:
            expand_spurions = (torch.cat(self.spurions)).expand(N,dinput)
            train_data = torch.cat((train_data,expand_spurions),dim = -1)
            self.input_size = train_data.shape[-1]
        
        
        train_dataset = TensorDataset(train_data,train_labels)
        self.train_dataset = train_dataset
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    def set_model(self,gens_list=gens_Lorentz,input_size = 4,init = "rand",equiv="False",rand="False",freeze = "False", activation = "ReLU",skip="False", hidden_size=10, n_hidden_layers=3):
        self.init = init
        self.equiv = equiv
        self.rand=rand
        self.freeze = freeze 
        self.activation = activation
        self.skip=skip
        self.input_size = input_size+sum([torch.numel(sp) for sp in self.spurions]) if (self.input_spurions == "True" or self.input_spurions==True) else input_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        
        

    def run_training(self,train_loader,nepochs = 1000,lam_vec = [0.0],seed = 98235, lr = 1e-3, opt = "Adam",symm_norm = "False"):    
        lam = lam_vec
        # Train the model, store train and test loss, print the loss every epoch
        train_loss = []
        symm_loss_vec = []
        tot_loss_vec = []
        running_loss = 0.0
        symm_loss = 0.0
        train_loss_lam = {}
        symm_loss_lam= {}
        tot_loss_lam = {}
        models = {}
        self.lr = lr
        self.opt = opt
        self.symm_norm = symm_norm
        train_loader_copy = copy.deepcopy(train_loader)

        if train_loader =="self":

            train_loader = self.train_loader
        
        self.train_seed = seed

        for lam_val in lam:
            train_loader_copy = copy.deepcopy(train_loader)
            np.random.seed(seed)
            torch.manual_seed(seed)
            #self.prepare_dataset()
            modelLorentz_symm = genNet(input_size = self.input_size, init = self.init ,equiv=self.equiv,rand=self.rand,freeze = self.freeze,activation = self.activation, skip = self.skip, n_hidden_layers = self.n_hidden_layers, hidden_size=self.hidden_size)

            model = modelLorentz_symm.to(devicef)
        
            # Define the loss function and optimizer
            if self.opt == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9)

            elif self.opt =="GD":
                optimizer = optim.GD(model.parameters(), lr=lr)
            
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)

            #losses
            criterion = nn.MSELoss()
            
            if symm_norm == "True":
                
                criterion_Lorentz = SymmLoss_norm(gens_list=self.gens_list, model = model)
                
            else:
                criterion_Lorentz = SymmLoss(gens_list=self.gens_list, model = model)
            
            
            ##############

            train_loss = []
            symm_loss_vec = []
            tot_loss_vec = []
            running_loss = 0.0
            symm_loss = 0.0
            start_MSE = 0.0
            end_MSE = 0.0
            start_symm = 0.0
            end_symm = 0.0
            deltat_MSE = 0.0
            deltat_symm = 0.0
            symm_loss_min = 10000000
            MSE_loss_min = 10000000
            tot_loss_min = 10000000
            
            for epoch in range(nepochs):
                model.train()
                running_loss = 0.0
                symm_loss = 0.0
                for i, data in enumerate(train_loader_copy):
                    inputs, labels = data
                    labels = torch.unsqueeze(labels.to(devicef),1)
                    inputs = inputs.to(devicef)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    start_MSE = time.time()
                    loss = criterion(outputs, labels)
                   
                    #loss_tot = loss+lam_val*loss_symm
                    #loss_tot.backward()
                    loss.backward()
                    #debugging
                    
                    # if (epoch % 100 == 0) and (i==len(train_loader)-1):
                    #     print("gradients MSE:")
                    #     for param in model.parameters():
                    #         print(param.grad)
                    #     # param.grad.zero_()
                    # # optimizer.zero_grad()
                    end_MSE = time.time()
                    deltat_MSE += end_MSE - start_MSE
                    start_symm = time.time()
                    loss_symm = criterion_Lorentz(input = inputs)
                    loss_symmtot = lam_val*loss_symm
                    loss_symmtot.backward()
                    end_symm = time.time()
                    deltat_symm += end_symm - start_symm
                    # if (epoch % 100 == 0 and (i==len(train_loader)-1)):
                    #     print(f"gradients MSE+{lam_val}*SYMM:")
                    #     for param in model.parameters():
                    #         print(param.grad)
                    # # print(f"gradients MSE+{lam_val}*SYMM:")
                    # # for param in model.parameters():
                    # #     print(param.grad)
                    ####
                    optimizer.step()
                    running_loss += loss.item()
                    symm_loss += loss_symm.item()
                                
                symm_loss_epoch = symm_loss / (1.0*len(train_loader_copy))
                running_loss_epoch = running_loss / (1.0*len(train_loader_copy))
                tot_loss_epoch = (lam_val*symm_loss+running_loss) / (1.0*len(train_loader_copy))
                
                
                train_loss.append(running_loss_epoch)
                symm_loss_vec.append(symm_loss_epoch)
                tot_loss_vec.append(tot_loss_epoch)
                
                
                # train_loss.append(running_loss / (1.0*len(train_loader_copy)))
                # symm_loss_vec.append(symm_loss / (1.0*len(train_loader_copy)))
                # tot_loss_vec.append((lam_val*symm_loss+running_loss) / (1.0*len(train_loader_copy)))
                
                #monitor
                if (epoch % 100 == 0) or (epoch == nepochs-1):
                    print(f"time for 1 epoch 1 batch MSE: {(deltat_MSE)/((epoch+1)*len(train_loader_copy))}, symm:{(deltat_symm)/((epoch+1)*len(train_loader_copy))}")
                    print(f"lambda = {lam_val} Epoch {epoch+1}, MSE loss: {train_loss[-1]:}, Lorentz loss: {symm_loss_vec[-1]:}")
                    if self.equiv =="True": 
                        print(f"bi-linear tensor layer:\n {model.bi_tensor}")
                        #print(f"bi-linear tensor grad:\n {model.bi_tensor.grad}")
                        if self.skip =="True":
                            print(f"skip layer: \n {model.skip_layer}")
                
                #save best model
                if  symm_loss_epoch < symm_loss_min:
                    symm_loss_min = symm_loss_epoch
                    self.models_best_symm[lam_val] = copy.deepcopy(model)
                if  running_loss_epoch < MSE_loss_min:
                    MSE_loss_min = running_loss_epoch
                    self.models_best_MSE[lam_val] = copy.deepcopy(model)
                if  tot_loss_epoch < tot_loss_min:
                    tot_loss_min = tot_loss_epoch
                    self.models_best_tot[lam_val] = copy.deepcopy(model)
                    
            
            self.train_loss_lam[lam_val] = train_loss
            self.symm_loss_lam[lam_val] = symm_loss_vec
            self.tot_loss_lam[lam_val] = tot_loss_vec
            
            model_clone = copy.deepcopy(model)
            self.models[lam_val] = model_clone
            
                    
                    


class analysis_trained(symm_net_train):

    def __init__(self):
        super().__init__()
        
    def get_trained(self, trained_net):
        self.__dict__ = {key:copy.deepcopy(value) for key, value in trained_net.__dict__.items()}
        
    
    def title(self):
        #text = f"N:{self.N} hidden size:{self.hidden_size} layers:{self.n_hidden_layers} activation:{self.activation} lr:{self.lr} opt:{self.opt}"
        text = f"toy N:{self.N} lr:{self.lr}"
        self.spurions_for_print = ""
        if self.broken_symm == "True" or self.broken_symm == True:
            text=f"{text} broken symm"
            spurions_for_print = "spurions:\n"
            text_spurions = f" "
            if self.input_spurions == "True" or self.input_spurions == True:
                text = f"{text} input spurions"
            for spurion in self.spurions_list:
                spurions_for_print+= f"{spurion}\n"
                spurion_text = f"{spurion}"
                #spurion_text = spurion_text
                text_spurions += f"spurion: {spurion_text}"
            self.spurions_for_print = spurions_for_print
            text = f"{text}{text_spurions}"
        if self.equiv== "True":
            text=f"{text} bi-linear layer"
            if self.skip =="True":
                text=f"{text} skip"
            if self.init=="eta" or self.init=="delta":
                text=f"{text} init: {self.init}"
            if self.freeze=="True":
                text=f"{text} freeze"
        if self.symm_norm == "True" or self.symm_norm == True:
                text = f"{text} norm"
        self.title_text = text
       
        self.filename = text.replace("[","").replace("]","").replace(",","_").replace(":","_").replace(" ","_")+f"_data_seed_{self.dataset_seed}_train_seed_{self.train_seed}"

        return text
    
    def save_trained(self,outdir = "./storage"):
        filename = self.title_text.replace(" ","_").replace(":","_")+f"data_seed_{self.dataset_seed}_train_seed_{self.train_seed}"
        with open(f'{outdir}/{filename}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self, f)

    def plot_losses(self,save = False, outdir = "./",filename = "",print_spurions = False):

        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_loss_lam
        symm_loss_lam = self.symm_loss_lam
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            plt.semilogy(range(len(train_loss_lam[lam_val])),train_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, MSE", color = color_vec[i%len(color_vec)])
            plt.semilogy(range(len(symm_loss_lam[lam_val])),symm_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, symm", color = color_vec[i%len(color_vec)],ls = "--")
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

    
    def plot_MSE_loss(self,save = False, outdir = "./",filename = "",print_spurions = False):
        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_loss_lam
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


        
            