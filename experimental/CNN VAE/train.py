import torch
import json
import torch.nn as nn
import random
from model import VAE
import torch.optim as optim

# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="train.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# setting up cuda and replication
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 999
logger.info(f'"Random Seed: ", {manualSeed}')
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

def weights_init(m):
    """
    A function to initialize model weights
    """
    classname = m.__class__.__name__
    if classname.find('Conv1D') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# VAE Loss Function
def vae_loss(reconstructed_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_divergence

if __name__ == "__main__":
    logger.info(device)
    # Load config
    with open('../../config.json', 'r') as file:
        config = json.load(file)

    prefix = "../../"

    # Training configs
    nz = config['training']['latent_dim']
    window_size = config['preprocessing']['window_size']
    num_epochs = config['training']['num_epochs']
    w_gan_training = False  # False always
    batch_size = config['training']['batch_size']
    lr = 0.0002
    in_dim = 1

    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    # Create the vae
    vae = VAE(nz, window_size)
    vae.to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    vae.apply(weights_init)

    # Print the model
    logger.info(vae)

    # data

    X_train = torch.load(f"X_train_{b_id}.pt")
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                             shuffle=True)

    optimizer = optim.Adam(vae.parameters(), lr=lr,betas=(0.5, 0.999))

    # training:

    logger.info("--vae--")
    logger.info("Starting Training Loop...")
    # Setup loss function
    # for epoch in range(num_epochs):
        # for i, x in enumerate(dataloader, 0):
        #     ############################
        #     # Update network: minimize  mse(x,A(x))
        #     ###########################
        #     vae.zero_grad()
        #     real = x.to(device).float()
        #     output = vae(real)
        #     err = criterion(output, real)
        #     err.backward()
        #     optimizer.step()
        

        # Print loss after every epoch
        # logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {err.item():.4f}')

    for epoch in range(num_epochs):
        total_loss = 0  # To accumulate loss for the current epoch
        for i, x in enumerate(dataloader,0):
            x = x.to(device).float()  # Move data to the correct device (GPU/CPU)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Forward pass: Get reconstructed output and latent variables
            reconstructed_x, mu, log_var = vae(x)

            # Compute VAE loss (reconstruction + KL divergence)
            loss = vae_loss(reconstructed_x, x, mu, log_var)

            # Backward pass: Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()  # Accumulate the total loss

        # Print loss after every epoch
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

    # save trained model
    torch.save(vae, f'vae_{b_id}.pth')
