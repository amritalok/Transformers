import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from model.EncoderLayer import EncoderLayer

class Encoder(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 8,
        dim_feedfordward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([ 
            EncoderLayer(dim_embedding, num_heads, dim_feedfordward, dropout) for _ in range(num_layers) 
        ])
    
    def forward(self, src:Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += positional_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
        return src