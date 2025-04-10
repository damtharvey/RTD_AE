# Dataset configuration
dataset_name = "COIL-20"
version = "d16"

# Model configuration
model_name = "RTD AutoEncoder H1"
input_dim = 128 * 128  # COIL-20 image size
latent_dim = 16
n_hidden_layers = 3
hidden_dim = 512

# Training configuration
max_epochs = 400
batch_size = 256
lr = 5e-4
accelerator = "cuda"

# RTD configuration
rtd_every_n_batches = 1
rtd_start_epoch = 0
rtd_l = 1.0  # RTD loss weight
card = 50  # Number of points in persistence diagram
n_threads = 50  # Number of threads for parallel computation
engine = "ripser"
is_sym = True 