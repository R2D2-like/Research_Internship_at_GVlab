'''
This code is for few-shot LfD. (step2)
input: (Sample, Action, Time step, 6) e.g. (6~8, 2, 200, 6)
output: (Sample, Time step, 3) e.g. (6~8, 2000, 3)
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from vae import VAE


class LfD(VAE):

    def __init__(self,
                 input_dim: int = 6,
                 output_dim: int = 3,
                 latent_dim: int = 5,
                 hidden_dim: int = None,
                 **kwargs) -> None:
        # Initialize the VAE with the same parameters
        super().__init__(input_dim=input_dim,
                         latent_dim=latent_dim,
                         hidden_dim=hidden_dim,
                         **kwargs)

        # Redefine the decoder to match the LfD output dimensions
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * 2000) # (B, 2000*hidden_dim)

        # Redefine the decoder layers to output the desired output_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.1)
        )

        # Adjust the final layer to output the correct dimension
        self.final_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim)
        )

    # The encode and reparameterize methods are inherited from VAE

    def decode(self, z: Tensor) -> Tensor:
        """
        Override the VAE's decode method to match the LfD.
        """
        result = self.decoder_input(z)  # (B, 2000*hidden_dim)
        result = result.view(-1, 2000, self.hidden_dim)  # (B, 2000, hidden_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Override the VAE's forward method to match the LfD.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Custom loss function for LfD.
        """
        recons = args[0]
        target = args[1]

        loss = F.mse_loss(recons, target)

        return {'loss': loss}

    # The sample and generate methods are inherited from VAE and can be used as is
