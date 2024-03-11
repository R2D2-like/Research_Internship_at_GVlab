import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lfd_baseline import LfD
import sys
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import *
import os

def load_data(exp_action_data_path, demo_data_path):
    exp_action_data = np.load(exp_action_data_path)
    demo_data = np.load(demo_data_path) 
    inputs, targets = None, None
    for sponge in TRAIN_SPONGES_LIST:
        if inputs is None:
            inputs = exp_action_data[sponge] #(DEMO_PER_SPONGE, 400, 6)
            targets = demo_data[sponge] #(DEMO_PER_SPONGE, 9, 2000)
            # (DEMO_PER_SPONGE, 9, 2000) -> (DEMO_PER_SPONGE, 2000, 9) 
            targets = np.transpose(targets, (0, 2, 1)) #(DEMO_PER_SPONGE, 2000, 9)
            # (DEMO_PER_SPONGE, 2000, 9) -> (DEMO_PER_SPONGE, 2000, 3)
            targets = targets[:, :, :3] #(DEMO_PER_SPONGE, 2000, 3)
        else:
            inputs = np.concatenate([inputs, exp_action_data[sponge]], axis=0) #(len(TRAIN_SPONGES_LIST), 400, 6)
            targets = np.concatenate([targets, demo_data[sponge]], axis=0) #(len(TRAIN_SPONGES_LIST), 2000, 3)
    return inputs, targets

def train(model, data_loader, optimizer, device,num_epochs=10000):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss_function(outputs, targets)['loss']
            loss.backward()
            optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")

def main():
    # load
    exp_action_data_path = '/root/Research_Internship_at_GVlab/real/step1/data/exploratory_action_preprocessed.npz'
    demo_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed.npz'
    encoder_weights_path = '/root/Research_Internship_at_GVlab/sim/model/vae_encoder.pth'

    # save
    dir = '/root/Research_Internship_at_GVlab/real/model/baseline/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_path = dir + 'baseline_model.pth'
    decoder_path = dir + 'baseline_decoder.pth'

    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data and create DataLoader
    inputs, targets = load_data(exp_action_data_path, demo_data_path)
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create model
    model = LfD(input_dim=6, output_dim=3, latent_dim=5, hidden_dim=32).to(device)
    
    # load encoder weights from pre-trained VAE
    model.load_state_dict(torch.load(encoder_weights_path), strict=False) # only load encoder weights
    
    # freeze encoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # set optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # train model
    train(model, data_loader, optimizer, device, num_epochs=10000)

    # save model
    torch.save(model.state_dict(), model_path)

    # save decoder weights
    decoder_weights = {k: v for k, v in model.state_dict().items() if 'decoder' in k}
    torch.save(decoder_weights, decoder_path)

if __name__ == "__main__":
    main()
