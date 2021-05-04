import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from model.SelfAttention import AttentionHead

class MultiHeadAttention(pl.LightningModule):
    def __init__(self, num_heads, dim_embedding, dim_k, dim_v):
        super().__init__()
        
        self.heads = nn.ModuleList(
            [AttentionHead(dim_embedding, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_embedding)
    
    def forward(self, w_q, w_k, w_v) -> Tensor:
        return self.linear( 
            torch.cat([h(w_q, w_k, w_v) for h in self.heads], dim=-1)
        )

        
    
    

