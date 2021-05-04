import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from torch import Tensor

class Residual(pl.LightningModule):
    def __init__(self, sublayer:nn.Module, dimension:int, dropout:float=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *tensors:Tensor) -> Tensor:
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

