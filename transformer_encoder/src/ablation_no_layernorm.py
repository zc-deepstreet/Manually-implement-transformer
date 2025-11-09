import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
import math

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_loader import load_wikitext2
from utils import save_checkpoint

# 使用相同的配置
BASE_CONFIG = {
    'd_model': 128,
    'num_heads': 4,
    'd_ff': 512,
    'num_layers': 2,
    'dropout': 0.1,
    'num_epochs': 20,
    'batch_size': 32,
    'seq_len': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'seed': 42,
    'save_interval': 10
}


class NoLayerNormTransformer(nn.Module):
    """无LayerNorm的Transformer"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1, pad_idx=0):
        super(NoLayerNormTransformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码
        self.positional_encoding = self._create_positional_encoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers - 无LayerNorm
        self.layers = nn.ModuleList([
            self._create_no_layernorm_layer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return torch.nn.Parameter(pe, requires_grad=False)

    def _create_no_layernorm_layer(self, d_model, num_heads, d_ff, dropout):
        """创建无LayerNorm的编码器层"""
        # 导入必要的组件
        from model import MultiHeadAttention, PositionWiseFFN

        class NoLayerNormEncoderLayer(nn.Module):
            def __init__(self, d_model, num_heads, d_ff, dropout):
                super(NoLayerNormEncoderLayer, self).__init__()
                # 保持多头注意力和前馈网络
                self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
                self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
                # 只有残差连接，没有LayerNorm
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, mask=None):
                # 无LayerNorm：直接使用原始输入进行残差连接
                attn_output = self.self_attention(x, x, x, mask)[0]
                x = x + self.dropout(attn_output)  # 残差连接，但没有LayerNorm

                ff_output = self.feed_forward(x)
                x = x + self.dropout(ff_output)  # 残差连接，但没有LayerNorm

                return x

        return NoLayerNormEncoderLayer(d_model, num_heads, d_ff, dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_padding_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        # Create padding mask
        mask = self.create_padding_mask(x)

        # Embed tokens with positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:x.size(1), :].transpose(0, 1)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        logits = self.output_layer(x)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AblationTrainer:
    def __init__(self, config, ablation_name, ablation_description):
        self.config = config
        self.ablation_name = ablation_name
        self.ablation_description = ablation_description
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置随机种子
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['seed'])

        # Load data
        self.train_loader, self.vocab, self.vocab_size = load_wikitext2(
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            max_sequences=2000
        )

        # Initialize ablation model
        self.model = NoLayerNormTransformer(
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

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Training history
        self.train_losses = []
        self.train_ppls = []
        self.learning_rates = []

        print(f"消融实验: {ablation_description}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Using device: {self.device}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_batches = 0

        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc="Training")):
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(src)
            logits = logits.view(-1, logits.size(-1))
            tgt = tgt.view(-1)
            loss = self.criterion(logits, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        self.scheduler.step()
        self.learning_rates.append(self.scheduler.get_last_lr()[0])

        avg_loss = total_loss / total_batches
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def train(self):
        print(f"开始训练: {self.ablation_description}")

        for epoch in range(self.config['num_epochs']):
            train_loss, train_ppl = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
                print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}")
                print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
                print("-" * 50)

        # 保存模型
        model_name = f"{self.ablation_name}_final.pt"
        torch.save(self.model.state_dict(), model_name)
        print(f"模型已保存: {model_name}")

    def get_results(self):
        return {
            'name': self.ablation_name,
            'description': self.ablation_description,
            'final_loss': self.train_losses[-1],
            'final_ppl': self.train_ppls[-1],
            'best_loss': min(self.train_losses),
            'best_ppl': min(self.train_ppls),
            'parameters': self.model.get_num_params(),
            'train_losses': self.train_losses,
            'train_ppls': self.train_ppls,
            'learning_rates': self.learning_rates
        }


def plot_training_curves(trainer, model_name):
    """绘制训练曲线"""
    # 确保目录存在
    os.makedirs('ablation_results', exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses)
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_ppls)
    plt.title(f'{model_name} - Training Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)

    plt.tight_layout()
    filename = f"ablation_results/{model_name}_training_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {filename}")


def save_results(results, model_name):
    """保存结果"""
    # 确保目录存在
    os.makedirs('ablation_results', exist_ok=True)

    filename = f"ablation_results/{model_name}_results.json"

    serializable_results = {
        'description': results['description'],
        'final_loss': float(results['final_loss']),
        'final_ppl': float(results['final_ppl']),
        'best_loss': float(results['best_loss']),
        'best_ppl': float(results['best_ppl']),
        'parameters': results['parameters']
    }

    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"结果已保存: {filename}")


def compare_with_baseline(ablation_results):
    """与基线模型比较"""
    # 加载基线结果
    try:
        baseline_path = 'ablation_results/baseline_results.json'
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)

        baseline_ppl = baseline_results['final_ppl']
        ablation_ppl = ablation_results['final_ppl']

        performance_drop = (ablation_ppl - baseline_ppl) / baseline_ppl * 100

        print("\n" + "=" * 60)
        print("消融实验对比分析")
        print("=" * 60)
        print(f"基线模型困惑度: {baseline_ppl:.4f}")
        print(f"消融模型困惑度: {ablation_ppl:.4f}")
        print(f"性能下降: {performance_drop:.2f}%")

        if performance_drop > 50:
            print("🔴 结论: LayerNorm是Transformer的关键组件!")
        elif performance_drop > 20:
            print("🟡 结论: LayerNorm对性能有显著影响")
        else:
            print("🟢 结论: LayerNorm的影响相对较小")

    except FileNotFoundError:
        print("警告: 未找到基线模型结果")


def run_no_layernorm_ablation():
    """运行无LayerNorm消融实验"""
    print("=" * 60)
    print("步骤 5: 无LayerNorm消融实验")
    print("=" * 60)

    # 确保结果目录存在
    os.makedirs('ablation_results', exist_ok=True)

    trainer = AblationTrainer(
        BASE_CONFIG,
        'no_layernorm',
        '无LayerNorm的Transformer'
    )
    trainer.train()
    results = trainer.get_results()

    # 绘制和保存结果
    plot_training_curves(trainer, 'no_layernorm')
    save_results(results, 'no_layernorm')

    # 与基线比较
    compare_with_baseline(results)

    print("\n" + "=" * 60)
    print("下一步：运行无前馈网络消融实验")
    print("请输入以下命令:")
    print("python src/ablation_no_ffn.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_no_layernorm_ablation()