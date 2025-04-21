import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder for VAE
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(256, latent_dim, int(window_size / (2 ** 3)), 1, 0, bias=False)
        )

        # Additional layers to output mean and log-variance
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


# Decoder for VAE
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, int(window_size / (2 ** 3)), 1, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x


# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(VAE, self).__init__()
        self.encoder = ConvEncoder(latent_dim, window_size)
        self.decoder = ConvDecoder(latent_dim, window_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # Random noise
        return mu + std * epsilon

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z.view(z.size(0), self.latent_dim, 1))  # Ensure z is in correct shape
        return x_reconstructed, mu, log_var


# VAE Loss Function
def vae_loss(reconstructed_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_divergence


if __name__ == "__main__":
    print("Variational Autoencoder")

    nz = 100
    w = 48
    X = torch.normal(0, 1, size=(303, 1, w))  # Example input

    model = VAE(nz, w)
    reconstructed_X, mu, log_var = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Reconstructed shape: {reconstructed_X.shape}")
    print(f"Mean shape: {mu.shape}")
    print(f"Log variance shape: {log_var.shape}")

    # Example VAE loss
    loss = vae_loss(reconstructed_X, X, mu, log_var)
    print(f"VAE loss: {loss.item()}")
