# configs/base.py

# Model configuration
d_model = 128
num_heads = 4
d_ff = 512
num_layers = 2
dropout = 0.1

# Training configuration
num_epochs = 50
batch_size = 32
seq_len = 64
learning_rate = 0.0003
weight_decay = 0.01
warmup_steps = 4000
max_grad_norm = 1.0

# Other configuration
seed = 42
save_interval = 10