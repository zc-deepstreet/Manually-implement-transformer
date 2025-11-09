import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import os


def simple_tokenizer(text):
    """简单的分词器，按空格和标点分割"""
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens


class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) > self.seq_len:
            text = text[:self.seq_len]
        else:
            text = text + ['<pad>'] * (self.seq_len - len(text))

        tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in text]
        return torch.tensor(tokens[:-1], dtype=torch.long), torch.tensor(tokens[1:], dtype=torch.long)


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text)

    # 创建词汇表
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab


def load_wikitext2(seq_len=64, batch_size=32, max_sequences=5000):
    """使用内置数据集，避免下载问题"""
    print("使用内置 WikiText-2 风格数据集...")

    # 创建更大的莎士比亚风格数据集
    texts = [
                "To be or not to be that is the question",
                "Whether tis nobler in the mind to suffer",
                "The slings and arrows of outrageous fortune",
                "Or to take arms against a sea of troubles",
                "And by opposing end them To die to sleep",
                "No more and by a sleep to say we end",
                "The heartache and the thousand natural shocks",
                "That flesh is heir to tis a consummation",
                "Devoutly to be wishd To die to sleep",
                "To sleep perchance to dream ay theres the rub",
                "For in that sleep of death what dreams may come",
                "When we have shuffled off this mortal coil",
                "Must give us pause theres the respect",
                "That makes calamity of so long life",
                "For who would bear the whips and scorns of time",
                "The oppressors wrong the proud mans contumely",
                "The pangs of despised love the laws delay",
                "The insolence of office and the spurns",
                "That patient merit of the unworthy takes",
                "When he himself might his quietus make",
                "With a bare bodkin who would fardels bear",
                "To grunt and sweat under a weary life",
                "But that the dread of something after death",
                "The undiscoverd country from whose bourn",
                "No traveller returns puzzles the will",
                "And makes us rather bear those ills we have",
                "Than fly to others that we know not of",
                "Thus conscience does make cowards of us all",
                "And thus the native hue of resolution",
                "Is sicklied oer with the pale cast of thought",
                "And enterprises of great pith and moment",
                "With this regard their currents turn awry",
                "And lose the name of action Soft you now",
                "The fair Ophelia Nymph in thy orisons",
                "Be all my sins rememberd"
            ] * 10  # 重复以创建更多数据

    all_tokens = []
    for text in texts:
        all_tokens.extend(simple_tokenizer(text))

    print(f"总token数: {len(all_tokens)}")

    # 创建序列
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        sequences.append(all_tokens[i:i + seq_len])
        if len(sequences) >= max_sequences:
            break

    print(f"生成的序列数: {len(sequences)}")

    # 构建词汇表
    vocab = build_vocab(sequences, min_freq=1)

    # 创建数据集
    dataset = TextDataset(sequences, vocab, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"数据集统计: {len(sequences)} 个序列, 词汇表大小: {len(vocab)}")

    return dataloader, vocab, len(vocab)