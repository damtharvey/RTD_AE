#!/usr/bin/env python3
"""
Autoencoder Training Script
This script implements the training of various autoencoder models including:
- Basic AutoEncoder
- Topological AutoEncoder
- RTD AutoEncoder
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import ripser
from persim import plot_diagrams

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rtd import RTDLoss, MinMaxRTDLoss, get_model
from src.utils import FromNumpyDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from collections import defaultdict

# Configuration
config = {
    "dataset_name": "COIL-20",
    "version": "d16",
    "model_name": "default",
    "max_epochs": 1,
    "accelerator": "cpu",  # Changed from gpus to accelerator for newer PyTorch Lightning
    "rtd_every_n_batches": 1,
    "rtd_start_epoch": 0,
    "rtd_l": 1.0,  # rtd loss 
    "n_runs": 1,  # number of runs for each model
    "card": 50,  # number of points on the persistence diagram
    "n_threads": 50,  # number of threads for parallel ripser computation
    "latent_dim": 16,  # latent dimension
    "input_dim": 128*128,
    "n_hidden_layers": 3,
    "hidden_dim": 512,
    "batch_size": 256,
    "engine": "ripser",
    "is_sym": True,
    "lr": 5e-4
}

def collate_with_matrix(batch):
    """Collate function that computes distance matrix on the fly."""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Convert data to tensor directly
    data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data])
    
    # Compute distance matrix
    dist_matrix = torch.cdist(data, data)
    
    # Convert labels to tensor with explicit dtype
    labels = torch.tensor(labels, dtype=torch.long)
    
    return data, dist_matrix, labels

def collate_with_matrix_geodesic(batch):
    """Collate function that uses pre-computed geodesic distances."""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Convert data to tensor directly
    data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data])
    
    # Convert labels to tensor with explicit dtype
    labels = torch.tensor(labels, dtype=torch.long)
    
    return data, labels

class RTDAutoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=5e-4, card=50, engine='ripser', is_sym=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.rtd_loss = RTDLoss(card=card, engine=engine, is_sym=is_sym)
        self.validation_step_outputs = []
        self.train_losses = []
        self.val_losses = []
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def training_step(self, batch, batch_idx):
        x, dist_matrix, _ = batch  # Unpack all three elements, ignore labels
        x_hat, z = self(x)
        
        # Compute reconstruction loss
        recon_loss = self.criterion(x_hat, x)
        
        # Compute distance matrix for latent space
        z_dist = torch.cdist(z, z)
        
        # Compute RTD loss
        rtd_loss = self.rtd_loss(dist_matrix, z_dist)
        
        # Total loss
        loss = recon_loss + rtd_loss
        
        # Log losses
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train_rtd_loss', rtd_loss, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def on_train_epoch_end(self):
        # Get the average training loss for this epoch
        avg_train_loss = self.trainer.callback_metrics.get('train_loss')
        if avg_train_loss is not None:
            self.train_losses.append(avg_train_loss.item())

    def validation_step(self, batch, batch_idx):
        x, dist_matrix, _ = batch  # Unpack all three elements, ignore labels
        x_hat, z = self(x)
        
        # Compute reconstruction loss
        recon_loss = self.criterion(x_hat, x)
        
        # Compute distance matrix for latent space
        z_dist = torch.cdist(z, z)
        
        # Compute RTD loss
        rtd_loss = self.rtd_loss(dist_matrix, z_dist)
        
        # Total loss
        loss = recon_loss + rtd_loss
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'val_recon_loss': recon_loss,
            'val_rtd_loss': rtd_loss,
            'val_loss': loss
        })
        
        return loss

    def on_validation_epoch_end(self):
        # Compute average losses
        avg_recon_loss = torch.stack([x['val_recon_loss'] for x in self.validation_step_outputs]).mean()
        avg_rtd_loss = torch.stack([x['val_rtd_loss'] for x in self.validation_step_outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # Log average losses
        self.log('val_recon_loss', avg_recon_loss)
        self.log('val_rtd_loss', avg_rtd_loss)
        self.log('val_loss', avg_loss)
        
        # Store validation loss
        self.val_losses.append(avg_loss.item())
        
        # Clear validation outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train_autoencoder(model, train_loader, val_loader=None, model_name='default', 
                     dataset_name='MNIST', accelerator='cpu', max_epochs=100, run=0, version=""):
    """Train a single autoencoder model."""
    version = f"{dataset_name}_{model_name}_{version}_{run}"
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), name='lightning_logs', version=version)
    trainer = pl.Trainer(
        logger=logger, 
        accelerator=accelerator,
        max_epochs=max_epochs, 
        log_every_n_steps=1, 
        num_sanity_val_steps=0
    )
    trainer.fit(model, train_loader, val_loader)
    return model

def load_data(dataset_name='COIL-20'):
    """Load the prepared dataset."""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, f"data/{dataset_name}/prepared")
    
    print(f"Loading data from: {data_dir}")
    
    # Check if prepared data exists
    if not os.path.exists(data_dir):
        print(f"Directory contents of {os.path.dirname(data_dir)}:")
        for root, dirs, files in os.walk(os.path.dirname(data_dir)):
            print(f"\nDirectory: {root}")
            for file in files:
                print(f"  {file}")
        raise ValueError(f"Prepared data directory {data_dir} does not exist")
    
    # Load the data
    train_data_path = os.path.join(data_dir, "train_data.npy")
    test_data_path = os.path.join(data_dir, "test_data.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")
    
    # Check which files exist
    missing_files = []
    for path in [train_data_path, test_data_path, train_labels_path, test_labels_path]:
        if not os.path.exists(path):
            missing_files.append(os.path.basename(path))
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        raise ValueError("Some prepared data files are missing")
    
    # Load the data
    print("Loading data files...")
    train_data = np.load(train_data_path).astype(np.float32)
    test_data = np.load(test_data_path).astype(np.float32)
    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)
    
    print(f"Loaded data:")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data, train_labels, test_labels

def visualize_reconstructions(model, test_data, num_samples=10, save_path=None):
    """Visualize original and reconstructed images."""
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        samples = test_data[indices]
        
        # Convert to tensor and move to device
        samples_tensor = torch.FloatTensor(samples).to(next(model.parameters()).device)
        
        # Get reconstructions
        reconstructions, _ = model(samples_tensor)
        
        # Convert to numpy
        samples = samples.reshape(-1, 128, 128)
        reconstructions = reconstructions.cpu().numpy().reshape(-1, 128, 128)
        
        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            axes[0, i].imshow(samples[i], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructions[i], cmap='gray')
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('Original')
        axes[1, 0].set_ylabel('Reconstructed')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def visualize_latent_space(model, test_data, test_labels, save_path=None):
    """Visualize the latent space using t-SNE."""
    model.eval()
    with torch.no_grad():
        # Get latent representations
        test_tensor = torch.FloatTensor(test_data).to(next(model.parameters()).device)
        _, latent = model(test_tensor)
        latent = latent.cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=test_labels, cmap='tab20')
        plt.colorbar(scatter)
        plt.title('Latent Space Visualization (t-SNE)')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def visualize_persistence_diagrams(test_data, save_path=None):
    """Visualize persistence diagrams for the test data."""
    # Compute distance matrix
    dist_matrix = np.zeros((len(test_data), len(test_data)))
    for i in range(len(test_data)):
        for j in range(i+1, len(test_data)):
            dist = np.linalg.norm(test_data[i] - test_data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Compute persistence diagrams
    diagrams = ripser.ripser(dist_matrix, maxdim=1)['dgms']
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plot_diagrams(diagrams[0], show=False)
    plt.title('H0 Persistence Diagram')
    
    plt.subplot(122)
    plot_diagrams(diagrams[1], show=False)
    plt.title('H1 Persistence Diagram')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load and prepare data
    dataset_name = config['dataset_name']
    train_data, test_data, train_labels, test_labels = load_data(dataset_name)
    
    # Create datasets
    train_dataset = FromNumpyDataset(train_data, train_labels)
    test_dataset = FromNumpyDataset(test_data, test_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=2,
        collate_fn=collate_with_matrix,
        shuffle=True
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=2,
        collate_fn=collate_with_matrix,
        shuffle=False
    )

    # Create model
    encoder, decoder = get_model(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        hidden_dim=config['hidden_dim']
    )

    model = RTDAutoencoder(
        encoder=encoder,
        decoder=decoder,
        lr=config['lr'],
        card=config['card'],
        engine=config['engine'],
        is_sym=config['is_sym']
    )

    # Create visualization directory
    vis_dir = os.path.join('visualizations', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(vis_dir, exist_ok=True)

    # Train model
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=config['model_name'],
        dataset_name=config['dataset_name'],
        accelerator=config['accelerator'],
        max_epochs=config['max_epochs'],
        run=config.get('n_runs', 0),
        version=config.get('version', '')
    )

    # Generate visualizations
    print("Generating visualizations...")
    
    # Visualize reconstructions
    visualize_reconstructions(
        trained_model,
        test_data,
        num_samples=10,
        save_path=os.path.join(vis_dir, 'reconstructions.png')
    )
    
    # Visualize latent space
    visualize_latent_space(
        trained_model,
        test_data,
        test_labels,
        save_path=os.path.join(vis_dir, 'latent_space.png')
    )
    
    # Visualize persistence diagrams
    visualize_persistence_diagrams(
        test_data,
        save_path=os.path.join(vis_dir, 'persistence_diagrams.png')
    )
    
    # Plot training curves
    plot_training_curves(
        trained_model.train_losses,
        trained_model.val_losses,
        save_path=os.path.join(vis_dir, 'training_curves.png')
    )

    # Save the trained model
    os.makedirs('trained_models', exist_ok=True)
    torch.save(trained_model.encoder.state_dict(), os.path.join('trained_models', 'encoder.pt'))
    torch.save(trained_model.decoder.state_dict(), os.path.join('trained_models', 'decoder.pt'))
    print(f"Saved trained model to trained_models/")
    print(f"Saved visualizations to {vis_dir}/")

if __name__ == "__main__":
    main() 