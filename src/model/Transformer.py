import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn import functional as f
from utils.Helper import *
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(pl.LightningModule):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 6,
        dim_feedfordward: int = 512,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers = num_encoder_layers,
            dim_embedding = dim_embedding,
            num_heads = num_heads,
            dim_feedfordward = dim_feedfordward,
            dropout = dropout
        )

        self.decoder = Decoder(
            num_layers = num_decoder_layers,
            dim_embedding = dim_embedding,
            num_heads = num_heads,
            dim_feedfordward = dim_feedfordward,
            dropout = dropout
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src:Tensor, tgt:Tensor) -> Tensor:
        print(f'src shape: {src.size()}, target_size: {tgt.size()}')
        return self.decoder(tgt, self.encoder(src))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f'x: {x}, y:{y}')
        y_hat = self.decoder(y, self.encoder(x))
        PAD_IDX = en_vocab.stoi['<pad>']
        loss = self.criterion(y_hat, y)
        self.log(f'training loss: {loss}')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
