import torch
import torch.nn as nn


# Autoencoder -----

class Autoencoder(nn.Module):
    def __init__(self, window_size):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten the input: (batch_size, 1, 48) -> (batch_size, 48)
            nn.Linear(window_size, 32),  # First dense layer: 48 -> 32
            nn.ReLU(True),               # ReLU activation

            nn.Linear(32, 16),           # Second dense layer: 32 -> 16
            nn.ReLU(True),               # ReLU activation

            nn.Linear(16, 8)             # Third dense layer: 16 -> 8 (latent space)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),            # First dense layer: 8 -> 16
            nn.ReLU(True),               # ReLU activation

            nn.Linear(16, 32),           # Second dense layer: 16 -> 32
            nn.ReLU(True),               # ReLU activation

            nn.Linear(32, window_size),  # Output layer: 32 -> 48
            nn.Tanh()                    # Ensure output is between -1 and 1
        )

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)  # Flatten and pass through encoder

        # Decode the latent representation
        x = self.decoder(x)  # Pass through decoder

        # Reshape back to (batch_size, 1, 48)
        x = x.view(-1, 1, 48)

        return x


# Main program to test the model
if __name__ == "__main__":
    window_size = 48  # The input size of the autoencoder
    model = Autoencoder(window_size)  # Initialize the autoencoder
    print(model)

    # Create a random input tensor of shape (batch_size, 1, 48)
    X = torch.normal(0, 1, size=(10, 1, window_size))  # 10 samples, each of size (1, 48)
    print(X.shape)
    Y = model(X)
    print(Y.shape)  # Should output (10, 1, 48), same as input shape
