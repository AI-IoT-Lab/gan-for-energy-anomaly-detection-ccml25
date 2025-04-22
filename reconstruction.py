import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time

def reconstruct(x, iters, netG, criterion=None, latent_dim=10):
    """
    Function to reconstruct multiple subsequences at a time.
    Parameters
    -------------
        X: pytorch tensor
          A  batch  of subsequences (batch_size, latent_dim, 1)
        iters: int
          Number of iterations for performing gradient descent in the noise space
        criterion:
          A differentiable reconstruction loss.
        latent_dim:
          Number of dimensions of the noise (latent) space
    Returns
    -----------
      Z: pytorch tensor
         The noise which reconstructs X.
      X_: pytorch tensor
         The reconstructed input
      loss_array:
         The final reconstruction loss between X and X_
    """
    loss_array = None
    x_ = None
    z = torch.zeros(size=(x.shape[0], latent_dim, 1), device=device)
    z.requires_grad = True
    optimizerz = torch.optim.Adam([z], lr=0.1)
    for i in range(iters):
        optimizerz.zero_grad()
        x_ = netG(z)
        # Compute the loss value
        loss_array = criterion(x.float(), x_)
        loss = loss_array.mean()
        loss.backward()
        optimizerz.step()

    return z, x_, loss_array
