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

class DecoderLayer(pl.LightningModule):
    def __init__(
        self,
        dim_embedding: int = 512,
        num_heads: int = 6,
        dim_feedfordward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        dim_k = dim_v = dim_embedding // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )

        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )

        self.feed_forward = Residual(
            feed_forward(dim_embedding, dim_feedfordward),
            dimension = dim_embedding,
            dropout = dropout
        )

    def forward(self, tgt:Tensor, memory:Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


