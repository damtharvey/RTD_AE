# RTD Autoencoder Training

This repository contains the training code for a Topologically Regularized Autoencoder using the RTD (Reconstruction Topological Distance) loss.

## Project Structure

```
├── src/
│   ├── rtd.py          # RTD loss implementation
│   └── utils.py        # Utility functions
├── data/               # Data directory
├── trained_models/     # Saved models
├── prepare_data.py    # Data preparation script
├── train_ae.py        # Main training script
├── config.py          # Configuration file
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Installation

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - First, run the data preparation script:
   ```bash
   python prepare_data.py
   ```
   This will:
   - Create necessary directories
   - Download and process the dataset
   - Save processed data in `data/{dataset_name}/prepared/` directory
   - Generate files: `train_data.npy`, `train_labels.npy`, `test_data.npy`, `test_labels.npy`

2. Configure the model:
   - Edit `config.py` to set your desired parameters:
     - Dataset name
     - Model architecture (input_dim, latent_dim, hidden_dim, etc.)
     - Training parameters (batch_size, learning rate, epochs)
     - RTD parameters (card, engine, etc.)

3. Train the model:
```bash
python train_ae.py
```

4. The trained model will be saved in `trained_models/` directory.

## Configuration

The main configuration parameters in `config.py` are:

- `dataset_name`: Name of the dataset
- `input_dim`: Input dimension
- `latent_dim`: Latent space dimension
- `hidden_dim`: Hidden layer dimension
- `n_hidden_layers`: Number of hidden layers
- `batch_size`: Training batch size
- `max_epochs`: Number of training epochs
- `lr`: Learning rate
- `card`: Number of points in persistence diagram
- `rtd_l`: RTD loss weight

## Features

- Autoencoder with configurable architecture
- RTD loss for topological regularization
- Support for different datasets
- Training and validation loops
- Model checkpointing
- TensorBoard logging

## License

This project is licensed under the MIT License - see the LICENSE file for details. 