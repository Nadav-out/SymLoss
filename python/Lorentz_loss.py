import torch
import torch.nn as nn
from torch.autograd import grad
import einops

class LorentzLoss(nn.Module):
    """
    LorentzLoss - A custom loss module to enforce Lorentz invariance/covariance of neural networks.
    
    This module calculates a loss term that penelizing diviations form Lorentz covariance.

    Attributes:
    -----------
    model : torch.nn.Module
        The neural network model whose output will be used to compute the loss.
    device : torch.device
        The device on which tensors will be allocated (e.g., CPU or GPU).
    lorentz_gens : torch.Tensor
        A tensor of shape [6, 4, 4] representing the Lorentz generators for rotations and boosts.

    Methods:
    --------
    forward(input, spin='scalar'):
        Computes the Lorentz invariant loss for the given input based on the specified spin type ('scalar' or 'vector').

    Parameters:
    -----------
    device : torch.device
        The device to use for creating necessary tensors.
    model : torch.nn.Module
        The model whose outputs are evaluated against Lorentz invariance.

    Example:
    --------
    >>> model = MyModel()
    >>> loss_module = LorentzLoss(device=torch.device('cuda'), model=model)
    >>> input_tensor = torch.randn(10, 8, device=torch.device('cuda'))  # Batch of 10, each with 8 features
    >>> loss = loss_module(input_tensor, spin='scalar')
    """
    def __init__(self, device, model):
        super(LorentzLoss, self).__init__()

        self.model = model
        self.device = device
        
        # Initialize Lorentz Generators
        self.lorentz_gens = torch.zeros([6, 4, 4], dtype=torch.float32, device=device)
        
        # Sigma matrix for rotations
        sig_mat = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=device)
        
        # Rotations using the sigma matrix
        self.lorentz_gens[0, [2, 2, 3, 3], [2, 3, 2, 3]] = sig_mat.flatten()
        self.lorentz_gens[1, [3, 3, 1, 1], [3, 1, 3, 1]] = sig_mat.flatten()
        self.lorentz_gens[2, [1, 1, 2, 2], [1, 2, 1, 2]] = sig_mat.flatten()

        # Boosts using the absolute values of the sigma matrix
        self.lorentz_gens[3, [0, 0, 1, 1], [0, 1, 0, 1]] = sig_mat.abs().flatten()
        self.lorentz_gens[4, [0, 0, 2, 2], [0, 2, 0, 2]] = sig_mat.abs().flatten()
        self.lorentz_gens[5, [0, 0, 3, 3], [0, 3, 0, 3]] = sig_mat.abs().flatten()

    def forward(self, input, spin='scalar'):
        """
        Forward pass to compute the Lorentz invariant loss.

        Parameters:
        -----------
        input : torch.Tensor
            The input tensor for which the loss needs to be computed.
        spin : str, optional
            The type of spin ('scalar' or 'vector') of the model's output (default is 'scalar').

        Returns:
        --------
        torch.Tensor
            The computed Lorentz invariant loss.

        Raises:
        -------
        ValueError
            If the `spin` argument is not 'scalar' or 'vector'.
        """
        if spin not in ['scalar', 'vector']:
            raise ValueError("Currently supporting only spin-0 ('scalar') and spin-1 ('vector') outputs")

        # Compute model output
        output = self.model(input)

        if spin == 'scalar':
            # Compute gradients with respect to input, shape [B, 4N], B is the batch size, N is the number of particles
            grads, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
            
            # Reshape grads to [B, N, d], where d=4 (4D space-time)
            grads = einops.rearrange(grads, 'B (N d) -> B N d', d=4)

            # Contract grads with generators, shape [6 (generators), B, N, d]
            gen_grads = torch.einsum('a c d, B N d->  a B N c ',self.lorentz_gens, grads)
            # Reshape to [6, B, (N d)]
            gen_grads = einops.rearrange(gen_grads, 'a B N d -> a B (N d)')

            # Dot with input [6 ,B]
            differential_trans = torch.einsum('a B N, B N -> a B', gen_grads, input)
            
            # Penilize any Lorentz violation
            scalar_loss = (differential_trans ** 2).sum()
            
            return scalar_loss
        
        elif spin == 'vector':
            # Pre-compute variation due to output vector using the Lorentz generators
            vector_variation = torch.einsum('B d,a c d->c B a', output, self.lorentz_gens) #shape is [4,B,4], for later simplicity 
            total_loss = 0

            # Process each component of the vector output
            for i in range(4):
                # Extract the i-th component for all batches
                output_component = output[:, i]
                
                # Compute gradients with respect to the i-th component of the output
                grads, = torch.autograd.grad(outputs=output_component, inputs=input, grad_outputs=torch.ones_like(output_component, device=self.device), create_graph=True)
                grads = einops.rearrange(grads, 'B (N d) -> B N d', d=4)

                # Compute the Lorentz invariant quantity and include the output vector variation
                big_tensor = torch.einsum('B c, a c d, B N d -> N B a', input, self.lorentz_gens, grads).sum(dim=0) - vector_variation[i] # shape [B,6]
                total_loss += (big_tensor ** 2).sum()

            return total_loss

        return None
