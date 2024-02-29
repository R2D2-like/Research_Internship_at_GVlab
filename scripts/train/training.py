import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lfd import LfD

def load_data(data_path):
    data = np.load(data_path)
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    targets = torch.tensor(data['targets'], dtype=torch.float32)
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
    data_path = 'path/to/your/data.npz'
    model_path = 'path/to/save/model.pth'
    encoder_weights_path = 'path/to/saved/encoder.pth'
    decoder_path = 'path/to/save/decoder.pth'

    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data and create DataLoader
    inputs, targets = load_data(data_path)
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
