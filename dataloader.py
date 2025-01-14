import tiktoken
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import math


class Tokenizer:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))


class ShakespeareDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, train: bool = True):
        self.tokenizer = tokenizer
        self.train = train

        # Read and tokenize the entire text
        with open(file_path, "r") as file:
            text = file.read()
        self.tokens = self.tokenizer.tokenize(text)

        # Calculate split point (80% train, 20% test)
        split_point = math.floor(len(self.tokens) * 0.8)

        # Select tokens based on train/test split
        if train:
            self.tokens = self.tokens[:split_point]
        else:
            self.tokens = self.tokens[split_point:]

    def __len__(self):
        return len(self.tokens) - 1  # -1 because we need input/target pairs

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.tokens[idx : idx + 1], self.tokens[idx + 1 : idx + 2]

    def collate_fn(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return input_ids, labels


# Create tokenizer
tokenizer = Tokenizer()

# Create train and test datasets
train_dataset = ShakespeareDataset("data/shakespeare.txt", tokenizer, train=True)
test_dataset = ShakespeareDataset("data/shakespeare.txt", tokenizer, train=False)

# Create dataloaders
batch_size = 32
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
)
