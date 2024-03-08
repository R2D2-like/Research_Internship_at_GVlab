from vae import VAE
import torch
import os
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_data(data_path):
    data = np.load(data_path)
    print("Loading dataset from", data_path)
    data = torch.tensor(data, dtype=torch.float)
    return data

def train(model, data_loader, optimizer, device, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, input, mu, log_var = model(data)
            loss_dict = model.loss_function(recon_batch, input, mu, log_var)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}, Reconstruction Loss: {loss_dict['Reconstruction_Loss']}, KLD: {loss_dict['KLD']}")

if __name__ == "__main__":
    data_path = "/root/Research_Internship_at_GVlab/sim/data/pre-processed_sim_data.npy"  # specify the path to data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data and create DataLoader
    data = load_data(data_path) # in paper (1000, 400, 6)
    dataset = TensorDataset(data, data) #TODO:using same data for x and t is correct?
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create model and set optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train model
    train(model, data_loader, optimizer, device)

    # save model
    torch.save(model.state_dict(), '/root/Research_Internship_at_GVlab/model/step1/vae_full.pth')
    torch.save(model.encoder.state_dict(), '/root/Research_Internship_at_GVlab/model/step1/vae_encoder.pth')
    torch.save(model.decoder.state_dict(), '/root/Research_Internship_at_GVlab/model/step1/vae_decoder.pth')