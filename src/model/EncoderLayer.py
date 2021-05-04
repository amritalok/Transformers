import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from model.MultiHeadAttention import MultiHeadAttention
from model.Residual import Residual

class EncoderLayer(pl.LightningModule):
    def __init__(
        self, 
        dim_embedding: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        dim_k = dim_v = dim_embedding // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )
        self.feed_forward = Residual(
            feed_forward(dim_embedding, dim_feedforward),
            dimension = dim_embedding,
            dropout = dropout
        )
    
    def forward(self, src:Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


