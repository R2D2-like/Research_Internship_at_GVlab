'''
This code is for pre-training haptic representation model. (step1)
input: (Sample, Time step, 6) e.g. (1000, 400, 6)
output: (Sample, Time step, 6) e.g. (1000, 400, 6)
'''
import torch
from torch import nn
from torch.nn import functional as F
from types import *
from typing import List
from torch import Tensor

class VAE(nn.Module):

    def __init__(self,
                 input_dim: int = 6,
                 latent_dim: int = 5,
                 hidden_dim: int = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dim is None:
            hidden_dim = 5 # TODO:not mentioned in the paper??
        self.hidden_dim = hidden_dim

        # Build Encoder (1 fully-connected layer)
        modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), #input should be batch✕time-steps✕6
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim*400, latent_dim) #(B, latent_dim)
        self.fc_var = nn.Linear(hidden_dim*400, latent_dim) #(B, latent_dim)


        # Build Decoder (1 fully-connected layer)
        modules = []

        # 1 fully-connected layer, relu activation with dropout rate 0.1
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim*400),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #flatten AFTER the fully connected layer not to lose time information
        result = torch.flatten(result, start_dim=1) #(B, hidden_dim*400)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result) # (B, latent_dim)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z) # (B, hidden_dim*400)
        result = result.view(-1, 400, self.hidden_dim) # (B, 400, hidden_dim)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var) # (B, latent_dim)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.06
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]