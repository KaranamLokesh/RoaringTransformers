"""
Data loading utilities for language modeling experiments.
Supports WikiText-2 via HuggingFace datasets, with a synthetic fallback.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticLMDataset(Dataset):
    """Synthetic language modeling dataset for quick testing."""

    def __init__(self, vocab_size, seq_len, num_samples=1000, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1), generator=gen)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]  # input, target


class WikiTextDataset(Dataset):
    """WikiText-2 language modeling dataset."""

    def __init__(self, seq_len, split='train', vocab_size=10000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        try:
            from datasets import load_dataset
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            text = ' '.join([t for t in ds['text'] if t.strip()])
        except Exception:
            # Fallback: generate pseudo-text
            text = ' '.join([f'word{i % vocab_size}' for i in range(100000)])

        # Simple character-level or word-level tokenization
        self.tokens = self._tokenize(text)

    def _tokenize(self, text):
        """Simple whitespace tokenizer with vocabulary capping."""
        words = text.split()
        vocab = {}
        idx = 2  # 0=pad, 1=unk
        tokens = []
        for w in words:
            if w not in vocab:
                if idx < self.vocab_size:
                    vocab[w] = idx
                    idx += 1
                else:
                    vocab[w] = 1  # UNK
            tokens.append(vocab[w])
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)])
        return chunk[:-1], chunk[1:]


class BPELMDataset(Dataset):
    """Real-corpus language modeling dataset with GPT-2 BPE tokenization.

    Loads WikiText-103 (or WikiText-2) via HuggingFace `datasets`, tokenizes
    the full split with the standard GPT-2 byte-level BPE (`tiktoken`), and
    chunks the concatenated token stream into non-overlapping blocks of
    `seq_len + 1`. Unlike the whitespace `WikiTextDataset`, this yields a
    real subword vocabulary so that validation perplexity reflects genuine
    language modeling rather than synthetic-data entropy.

    `max_tokens` caps the number of tokens used (useful for a budget GPU run).
    """

    GPT2_VOCAB_SIZE = 50257

    def __init__(self, seq_len, config='wikitext-103-raw-v1', split='train',
                 max_tokens=None):
        self.seq_len = seq_len
        self.vocab_size = self.GPT2_VOCAB_SIZE

        import tiktoken
        from datasets import load_dataset

        enc = tiktoken.get_encoding('gpt2')
        ds = load_dataset('Salesforce/wikitext', config, split=split)
        text = '\n'.join(t for t in ds['text'] if t.strip())
        ids = enc.encode_ordinary(text)
        if max_tokens is not None:
            ids = ids[:max_tokens]
        self.tokens = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def get_dataloader(dataset_type='synthetic', vocab_size=10000, seq_len=512,
                   batch_size=16, num_samples=1000, split='train',
                   config='wikitext-103-raw-v1', max_tokens=None):
    """Create a DataLoader for experiments.

    dataset_type:
        'synthetic'   -> random tokens (legacy, for the toy convergence run)
        'wikitext'    -> WikiText-2, whitespace tokenizer (legacy)
        'bpe'         -> WikiText-103/2 with GPT-2 BPE (recommended, real corpus)
    """
    if dataset_type == 'bpe':
        dataset = BPELMDataset(seq_len, config=config, split=split,
                               max_tokens=max_tokens)
    elif dataset_type == 'wikitext':
        dataset = WikiTextDataset(seq_len, split=split, vocab_size=vocab_size)
    else:
        dataset = SyntheticLMDataset(vocab_size, seq_len, num_samples=num_samples)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                      drop_last=True)
