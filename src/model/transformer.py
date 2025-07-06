import torch
from torch import nn
from torch import Tensor
from enum import Enum
from typing import Optional

from .utils import ActivationType

class EmbedType(Enum):
    START = 0
    PATCH = 1
    END = 2
    CLS = 3

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: ActivationType = ActivationType.GELU
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # input shape: (batch_size, seq_len, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == ActivationType.RELU:
            self.activation_fn = nn.ReLU()
        elif activation == ActivationType.GELU:
            self.activation_fn = nn.GELU()
        elif activation == ActivationType.LEAKY_RELU:
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(
        self,
        x: Tensor,  # shape: (batch_size, seq_len, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (seq_len, seq_len)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, seq_len)
    ) -> Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # attn_output shape: (batch_size, seq_len, d_model)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.linear2(self.dropout(self.activation_fn(self.linear1(x))))
        # ff_output shape: (batch_size, seq_len, d_model)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x  # shape: (batch_size, seq_len, d_model)
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.RELU
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,  # shape: (batch_size, seq_len, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (seq_len, seq_len)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, seq_len)
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x  # shape: (batch_size, seq_len, d_model)

class BinocularformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.RELU
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(len(EmbedType), d_model, padding_idx=0)
        # shape: (vocab_size=4, d_model)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

    def forward(
        self,
        x: Tensor,  # shape: (batch_size, seq_len, d_model)
        src_mask: Optional[Tensor] = None,  # shape: (seq_len, seq_len)
        src_key_padding_mask: Optional[Tensor] = None  # shape: (batch_size, seq_len)
    ) -> Tensor:
        bs, seq_len, _ = x.shape
        device = x.device

        start_token = self.embedding(torch.full((bs, 1), EmbedType.START.value, device=device))
        end_token = self.embedding(torch.full((bs, 1), EmbedType.END.value, device=device))
        cls_token = self.embedding(torch.full((bs, 1), EmbedType.CLS.value, device=device))
        x = x + self.embedding(torch.full((bs, seq_len), EmbedType.PATCH.value, device=device))
        x = torch.cat([start_token, x, end_token, cls_token], dim=1)
        
        # shape: (batch_size, seq_len + 3, d_model)
        return self.encoder(x, src_mask, src_key_padding_mask)
