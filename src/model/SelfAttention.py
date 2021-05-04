import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *

class AttentionHead(pl.LightningModule):
    def __init__(self, d_embedding:int, d_key:int, d_value:int):
        super().__init__()
        self.q = nn.Linear(d_embedding, d_key)
        self.k = nn.Linear(d_embedding, d_key)
        self.v = nn.Linear(d_embedding, d_value)
    
    def forward(self, w_q, w_k, w_v) -> Tensor:
        Q, K, V = self.q(w_q), self.k(w_k), self.v(w_v)
        return scaled_dot_product_attention(Q, K, V)
    


