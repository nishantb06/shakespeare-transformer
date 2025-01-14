import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import Transformer
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    n_heads: int = 12
    dim: int = 768
    max_seq_length: int = 2048
    num_layers: int = 12
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_dim = config.dim
        self.config = config
        self.n_heads = self.config.n_heads
        self.dropout = self.config.dropout

        self.qkv_weights = nn.Linear(self.n_dim, self.n_dim * 3)
        self.projection_layer = nn.Linear(self.n_dim, self.n_dim)

        self.attention_dropout = nn.Dropout(self.dropout)
        self.projection_dropout = nn.Dropout(self.dropout)

    # dimension of x to be (batch_size, token_length, dim) => (B, C, D)
    # return a tensor of similar shape but with attention scores weighted averaged value vectors
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv_weights(x).split(
            self.n_dim, dim=-1
        )  # each of q,k v have a shape of (B,C,D)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, N_HEADS , T , D // N_HEADS) => (B, 12, T , 64)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attention_matrix = (q @ k.transpose(-2, -1)) * (1 / q.size(-1) ** 0.5)
        attention_matrix = F.softmax(attention_matrix, dim=-1)  # (B, N_HEADS, T, T)
        attention_matrix = self.attention_dropout(
            attention_matrix
        )  # (B, N_HEADS, T, T)

        v = (
            (attention_matrix @ v)
            .transpose(2, 1)
            .contiguous()
            .view(
                B, T, C
            )  # the contiguous is important because , to make memory contiguous
        )  # torch.Size([1, 32, 768]) [B, T, C]
        v = self.projection_layer(v)
        v = self.projection_dropout(v)
        return v


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, 4 * config.dim)  # [n_embd, 4 * n_embd]
        self.c_proj = nn.Linear(4 * config.dim, config.dim)  # [4 * n_embd, n_embd]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)  # [B, T, 4 * n_embd]
        x = F.gelu(x)  # [B, T, 4 * n_embd]
        x = self.c_proj(x)  # [B, T, n_embd]
        x = self.dropout(x)  # [B, T, n_embd]
        return x


# combine MHA with Layer Norms
# ln1 -> attention -> ln2 -> feed forward
class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config=config)
        self.mlp = FeedForward(config=config)
        self.ln1 = nn.LayerNorm(self.config.dim)
        self.ln2 = nn.LayerNorm(self.config.dim)

    def forward(self, x) -> None:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# makes multiple layers and attaches the feed forward layer for final token prediction
# init weights , that is a empirical method to initialise weights.
# need to define
# 1. Embedding layer
# 2. Positional embedding layer
# 3. List of transformer blocks
# 4. final fully connected layer with output the size of vocabulary
# 5. layer norms and dropout
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config=Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.positional_embedding = nn.Embedding(
            self.config.vocab_size, self.config.dim
        )
        self.dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(self.config.dim)
        self.classificatiion_layer = nn.Linear(self.config.dim, self.config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx) -> None:
        B, T = idx.size()  # (Batch size, number of tokens) # (1, T)

        # get positional embedding for seq 1 to T
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(
            0
        )  # [1, T]

        token_embeddings = self.embedding(idx)  # (1, T, n_dim)
        positional_embeddings = self.positional_embedding(positions)

        x = self.dropout(token_embeddings + positional_embeddings)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)  # [B, T, n_embd]

        # final layer norm and linear projection for classification
        x = self.final_layer_norm(x)  # [B, T, n_embd]
        logits = self.classificatiion_layer(x)  # [B, T, vocab_size]
        # [You]     ========> are
        # [You are]   ======> a
        # [You are a]   ====> fool
        # [You are a fool] => EOS

        return logits


if __name__ == "__main__":
    config = Config()
    model = DecoderOnlyTransformer(config)

    # Example usage
    batch_size = 4
    seq_len = 128

    # Generate random input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Output shape:", logits.shape)

# Input shape: torch.Size([4, 128])
# Output shape: torch.Size([4, 128, 50257])