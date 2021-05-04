import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from model.DecoderLayer import DecoderLayer

class Decoder(pl.LightningModule):
    def __init__(
        self, 
        num_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 8,
        dim_feedfordward: int = 2048,
        dropout: int = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([ 
            DecoderLayer(dim_embedding, num_heads, dim_feedfordward, dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_embedding, dim_embedding)
    

    def forward(self, tgt:Tensor, memory:Tensor) -> Tensor:
        _, seq_len, dim_embedding = tgt.size()
        tgt += positional_encoding(seq_len, dim_embedding)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        
        return torch.softmax(self.linear(tgt), dim=-1)
