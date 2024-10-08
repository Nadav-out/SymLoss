import torch
import torch.nn as nn
import contextlib
import warnings




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
    
    
    
    
def symm_loss_scalar9(model, X, X_cartesian, jet_vec, mask=None, train=False, generators=None, take_mean = False, dict_vars = {"logE":0, "logPt":1, "eta":2, "phi":3, "log_R_E":4, "log_R_pt":5, "dEta":6, "dPhi":7, "dR":8}):
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
    
    dici = trans_dict
    dici["dEta"] = trans_dict_jet['eta'] - trans_dict['eta']
    dici["dPhi"] = trans_dict_jet['phi'] - trans_dict['phi']
    dici["log_R_pt"] = (trans_dict['logPt'] - trans_dict_jet['logPt'])
    dici["log_R_E"] = (trans_dict['logE'] - trans_dict_jet['logE'])
    K_tensor = (input[:,:,dict_vars["dEta"]].unsqueeze(-1) * (trans_dict['eta'] - trans_dict_jet['eta']))
    K_tensor += (input[:,:,dict_vars["dPhi"]].unsqueeze(-1) * (trans_dict['phi'] - trans_dict_jet['phi']))
    K_tensor /= input[:,:,dict_vars["dR"]].unsqueeze(-1)+1e-10 # Avoid division by zero
    dici["dR"] = K_tensor
    
    
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



