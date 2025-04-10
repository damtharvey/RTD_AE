import torch
from torch.utils.data import Dataset
import numpy as np

class FromNumpyDataset(Dataset):
    """Dataset class for loading numpy arrays."""
    def __init__(self, data, labels=None, compute_geodesic=False):
        self.data = data
        self.labels = labels if labels is not None else np.zeros(len(data))
        self.compute_geodesic = compute_geodesic
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if self.compute_geodesic:
            return self.data[idx], self.labels[idx], self.data[idx]
        return self.data[idx], self.labels[idx]

def compute_distance_matrix(x):
    """Compute pairwise distance matrix for a batch of data."""
    return torch.cdist(x, x)

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

def save_model(model, path):
    """Save model state dict."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model state dict."""
    model.load_state_dict(torch.load(path))
    return model

def get_latent_representations(encoder, dataloader, device='cpu'):
    """Extract latent representations from a model using a dataloader."""
    encoder.eval()
    latent_representations = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # data, dist_matrix, labels
                data, _, batch_labels = batch
            else:  # data, labels
                data, batch_labels = batch
                
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            else:
                data = torch.tensor(data).to(device)
                
            latent = encoder(data)
            latent_representations.append(latent.cpu().numpy())
            labels.append(batch_labels.numpy())
            
    return np.vstack(latent_representations), np.concatenate(labels) 