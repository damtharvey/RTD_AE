#!/usr/bin/python3
import os
os.environ['MallocStackLogging'] = '0'
os.environ['MallocStackLoggingNoCompact'] = '0'
os.environ['MallocErrorAbort'] = '0'
os.environ['MallocLogFile'] = '/dev/null'
import numpy as np
import torch
import torch.nn as nn
import ripser
from torch.utils.data import Dataset, DataLoader

def lp_loss(a, b, p=2):
    return (torch.sum(torch.abs(a-b)**p))

def get_indicies(DX, dgm, dim, card):
    # Sort points by persistence
    pers = dgm[:, 1] - dgm[:, 0]
    perm = np.argsort(pers)[::-1]  # Sort in descending order
    
    # Get the top 'card' points
    top_indices = perm[:card]
    
    # For each point, find its nearest neighbors
    indices = []
    for idx in top_indices:
        # Find the two points that form this persistence pair
        birth_time = dgm[idx, 0]
        death_time = dgm[idx, 1]
        
        # Find points in the distance matrix that have these times
        birth_point = np.argmin(np.abs(DX - birth_time))
        death_point = np.argmin(np.abs(DX - death_time))
        
        indices.append([birth_point, death_point])
    
    # Convert to numpy array and ensure it has the right shape
    indices = np.array(indices)
    if len(indices) == 0:
        indices = np.zeros((card, 2), dtype=np.int64)
    elif len(indices) < card:
        # Pad with zeros if needed
        padding = np.zeros((card - len(indices), 2), dtype=np.int64)
        indices = np.vstack([indices, padding])
    
    return indices

def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1
        
    if engine == 'ripser':
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0  # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))  # zero out diagonal
        
        # Use ripser to compute persistent homology
        dgms = ripser.ripser(DX_, maxdim=dim, coeff=2)['dgms']
        
        # Convert to the format expected by the rest of the code
        rc = {'dgms': dgms, 'pairs': []}
        
        # Create a dummy pairs list since ripser doesn't provide it
        for d in range(dim+1):
            if d < len(dgms):
                rc['pairs'].append([(0, 0) for _ in range(len(dgms[d]))])
            else:
                rc['pairs'].append([])
    
    all_indicies = []  # for every dimension
    for d in range(1, dim+1):
        if d < len(rc['dgms']) and len(rc['dgms'][d]) > 0:
            all_indicies.append(get_indicies(DX, rc['dgms'][d], d, card))
        else:
            # If no features in this dimension, return zeros
            all_indicies.append([0] * (4*card))
    return all_indicies

class RTD_differentiable(nn.Module):
    def __init__(self, card=50, engine='ripser', is_sym=True):
        super().__init__()
        self.card = card
        self.engine = engine
        self.is_sym = is_sym

    def forward(self, DX, DY, immovable=0):
        # Ensure distance matrices are symmetric
        if self.is_sym:
            DX = (DX + DX.t()) / 2
            DY = (DY + DY.t()) / 2

        # Normalize distance matrices
        DX = DX / DX.max()
        DY = DY / DY.max()

        # Convert to numpy for ripser
        DX_np = DX.detach().cpu().numpy()
        DY_np = DY.detach().cpu().numpy()

        # Compute persistence diagrams
        if self.engine == 'ripser':
            dgm_X = ripser.ripser(DX_np, maxdim=0, distance_matrix=True)['dgms'][0]
            dgm_Y = ripser.ripser(DY_np, maxdim=0, distance_matrix=True)['dgms'][0]
        else:
            raise ValueError(f"Engine {self.engine} not supported")

        # Sort points by persistence and select top card points
        if len(dgm_X) > 0:
            dgm_X = dgm_X[np.argsort(dgm_X[:, 1] - dgm_X[:, 0])[::-1][:self.card]]
        if len(dgm_Y) > 0:
            dgm_Y = dgm_Y[np.argsort(dgm_Y[:, 1] - dgm_Y[:, 0])[::-1][:self.card]]

        # Get indices for the selected points
        idx_X = get_indicies(DX_np, dgm_X, dim=0, card=self.card)
        idx_Y = get_indicies(DY_np, dgm_Y, dim=0, card=self.card)

        # Ensure indices are within bounds
        n = DX.shape[0]
        idx_X = np.clip(idx_X, 0, n-1)
        idx_Y = np.clip(idx_Y, 0, n-1)

        # Convert indices to tensors
        idx_X = torch.from_numpy(idx_X).to(DX.device)
        idx_Y = torch.from_numpy(idx_Y).to(DY.device)

        # Compute RTD
        if immovable == 0:
            return torch.mean(torch.abs(DX[idx_X[:, 0], idx_X[:, 1]] - DY[idx_X[:, 0], idx_X[:, 1]]))
        elif immovable == 1:
            return torch.mean(torch.abs(DX[idx_Y[:, 0], idx_Y[:, 1]] - DY[idx_Y[:, 0], idx_Y[:, 1]]))
        else:
            return torch.mean(torch.abs(DX[idx_X[:, 0], idx_X[:, 1]] - DY[idx_X[:, 0], idx_X[:, 1]])) + \
                   torch.mean(torch.abs(DX[idx_Y[:, 0], idx_Y[:, 1]] - DY[idx_Y[:, 0], idx_Y[:, 1]]))

class RTDLoss(nn.Module):
    def __init__(self, card=50, engine='ripser', is_sym=True):
        super().__init__()
        self.card = card
        self.engine = engine
        self.is_sym = is_sym
        self.rtd = RTD_differentiable(card=card, engine=engine, is_sym=is_sym)
        
    def forward(self, x_dist, z_dist):
        # Ensure distance matrices are symmetric
        x_dist = (x_dist + x_dist.t()) / 2
        z_dist = (z_dist + z_dist.t()) / 2
        
        # Normalize distance matrices
        x_dist = x_dist / x_dist.max()
        z_dist = z_dist / z_dist.max()
        
        # Compute RTD loss
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        rtd_zx = self.rtd(z_dist, x_dist, immovable=1)
        
        return (rtd_xz + rtd_zx) / 2
    
class MinMaxRTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='ripser', is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.p = lp
        self.rtd_min = RTD_differentiable(dim, card, 'minimum', n_threads, engine)
        self.rtd_max = RTD_differentiable(dim, card, 'maximum', n_threads, engine)
    
    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd_min(x_dist, z_dist, immovable=1) + self.rtd_max(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd_min(z_dist, x_dist, immovable=2) + self.rtd_max(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz): # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss

def get_model(input_dim, latent_dim, n_hidden_layers=3, hidden_dim=512):
    """Create encoder and decoder models for the autoencoder."""
    # Encoder
    encoder_layers = []
    in_dim = input_dim
    
    # First layer
    encoder_layers.extend([
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    ])
    in_dim = hidden_dim
    
    # Hidden layers with constant size
    for _ in range(n_hidden_layers - 1):
        encoder_layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ])
    
    # Final layer to latent space
    encoder_layers.append(nn.Linear(in_dim, latent_dim))
    
    encoder = nn.Sequential(*encoder_layers)
    
    # Decoder
    decoder_layers = []
    in_dim = latent_dim
    
    # First layer
    decoder_layers.extend([
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    ])
    in_dim = hidden_dim
    
    # Hidden layers with constant size
    for _ in range(n_hidden_layers - 1):
        decoder_layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ])
    
    # Final layer to input space
    decoder_layers.append(nn.Linear(in_dim, input_dim))
    
    decoder = nn.Sequential(*decoder_layers)
    
    return encoder, decoder 