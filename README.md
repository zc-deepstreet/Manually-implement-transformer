# Manually-implement-transformer
This project demonstrates how to manually build a transformer and train it on WikiText-2 and Tiny Shakespeare. It is suitable for learning and understanding transformers.
Below is the training loss.
<img width="1200" height="400" alt="2df33d4aead142e391b82bcf139f4df5" src="https://github.com/user-attachments/assets/6aae8bb2-cb55-4f76-858f-d0a01e2e1d2f" />

The project structure is as follows:
```
transformer_encoder/
├── configs/
│   ├── base.py
│   ├── results/
│   └── scripts/
├── run.sh
├── src/
│   ├── ablation_results/
│   ├── results/
│   ├── __init__.py
│   ├── ablation_no_layernorm.py
│   ├── ablation_no_positional.py
│   ├── ablation_no_residual.py
│   ├── ablation_single_head.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── training_curves.png
│   ├── utils.py
│   └── wikitext-2-v1.zip
├── requirements.txt
└── README.md
```

'seed': 42,

Requirement：
torch>=1.9.0
torchtext>=0.10.0
numpy>=1.21.0
matplotlib>=3.3.0
tqdm>=4.62.0
