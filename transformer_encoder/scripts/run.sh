#!/bin/bash

# Set up environment
echo "Setting up environment..."
conda create -n transformer python=3.10 -y
conda activate transformer

# Install dependencies
echo "Installing dependencies..."
pip install torch torchtext numpy matplotlib tqdm pyyaml

# Create necessary directories
mkdir -p results
mkdir -p checkpoints

# Run training
echo "Starting training..."
python train.py

echo "Training completed!"