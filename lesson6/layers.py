import math
from copy import deepcopy
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, pad_index: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        self.position = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head: int):
        super().__init__()
        self.scale = math.sqrt(d_head)

    def forward(self, query, key, value, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, weights = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attention = deepcopy(mha)
        self.ffn = deepcopy(ffn)
        self.norm1 = nn.LayerNorm(mha.d_model)
        self.norm2 = nn.LayerNorm(mha.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attention(x, x, x, mask)[0]
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        x = self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x