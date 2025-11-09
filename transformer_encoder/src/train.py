import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import math

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入当前目录的模块
from model import TransformerEncoder
from data_loader import load_wikitext2
from utils import save_checkpoint

# 配置参数
config = {
    # Model configuration
    'd_model': 128,
    'num_heads': 4,
    'd_ff': 512,
    'num_layers': 2,
    'dropout': 0.1,

    # Training configuration
    'num_epochs': 50,
    'batch_size': 32,
    'seq_len': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,

    # Other configuration
    'seed': 42,
    'save_interval': 10
}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        self.train_loader, self.vocab, self.vocab_size = load_wikitext2(
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            max_sequences=2000
        )

        # Initialize model
        self.model = TransformerEncoder(
            vocab_size=self.vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_len=config['seq_len'],
            dropout=config['dropout']
        ).to(self.device)

        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 使用步长学习率调度
        self.scheduler = StepLR(
            self.optimizer,
            step_size=15,
            gamma=0.5
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Training history
        self.train_losses = []
        self.train_ppls = []

        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Using device: {self.device}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_batches = 0

        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc="Training")):
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(src)

            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            tgt = tgt.view(-1)

            loss = self.criterion(logits, tgt)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        # 每个epoch结束后更新学习率
        self.scheduler.step()

        avg_loss = total_loss / total_batches
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def train(self):
        print("Starting training...")

        for epoch in range(self.config['num_epochs']):
            train_loss, train_ppl = self.train_epoch()

            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)

            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)

            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_ppl': train_ppl,
                    'vocab': self.vocab
                }
                save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model
        torch.save(self.model.state_dict(), 'final_model.pt')

    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_ppls)
        plt.title('Training Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train()
    trainer.plot_training_curves()


if __name__ == "__main__":
    main()