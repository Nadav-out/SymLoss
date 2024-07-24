import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

class SymmLoss_pT_eta_phi(nn.Module):

    def __init__(self,model, gens_list = ["Lx", "Ly", "Lz", "Kx", "Ky", "Kz"],device = devicef):
        super(SymmLoss_pT_eta_phi, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        
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
        
        

    def forward(self, input, model_rep='scalar',norm = "none",nfeatures = "",eta_linlog = "lin",phi_linlog = "lin"):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)
        if nfeatures!="":
            dim = nfeatures
        else:
            dim = 4 #self.generators.shape[-1]
        #Assuming input is shape [B,d*N] d is the number of features, N is the number of particles
        input_reshaped = einops.rearrange(input, '... (N d) -> ... N d',d = dim)
        
        E = input[:,0::dim] #assuming input features are ordered as (E,pT,eta,phi)
        
        pT = input[:,1::dim]
        
        eta = input[:,2::dim]
        if eta_linlog == "log":
            eta = torch.exp(eta)
        
        phi = input[:,3::dim]
        if phi_linlog == "log":
            phi = torch.exp(phi)
        
        
        GenList = self.generators  
        
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
            varsE[i] = dE[GenList[i]]
            varspT[i] = dpT[GenList[i]]
            varseta[i] = deta[GenList[i]]/eta if eta_linlog == "log" else deta[GenList[i]]
            varsphi[i] = dphi[GenList[i]]/phi if phi_linlog == "log" else dphi[GenList[i]]
        
        varsSymm = torch.stack((varsE,varspT,varseta,varsphi), dim = -1) #[n,B,N,d]
        #print(varsSymm.shape)
            
        # Compute model output, shape [B]
        output = self.model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads_input, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
        
        # Reshape grads to [B, N, d] 
        grads_input = einops.rearrange(grads_input, '... (N d) -> ... N d',d = dim)

            
        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', varsSymm, grads_input)
        
        scalar_loss = (differential_trans ** 2).mean()
            
            #add norm part here?
     
            
        return scalar_loss
