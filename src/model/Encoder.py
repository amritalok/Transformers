import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from model.EncoderLayer import EncoderLayer
from typing import Optional

class Encoder(pl.LightningModule):
    """
    Encoder module of the Transformer architecture as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    The Encoder consists of a stack of identical layers, each containing:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    
    The encoder processes the input sequence and produces representations that are used by the decoder.
    Each position in the encoder can attend to all positions in the previous layer of the encoder.
    """
    def __init__(
        self,
        num_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 8,
        dim_feedfordward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize an Encoder instance.
        
        Args:
            num_layers: Number of encoder layers (default: 6)
            dim_embedding: Dimension of the input embeddings (default: 512)
            num_heads: Number of attention heads (default: 8)
            dim_feedfordward: Dimension of the feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Stack of identical encoder layers
        self.layers = nn.ModuleList([ 
            EncoderLayer(dim_embedding, num_heads, dim_feedfordward, dropout) for _ in range(num_layers) 
        ])
    
    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Process input sequence through the encoder stack.
        
        The forward pass consists of processing the input embeddings (with positional encoding
        already added) through each encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim_embedding] with positional encoding
            
        Returns:
            Processed tensor of the same shape as input
        """
        # Process through each encoder layer
        encoder_output = x
        for layer in self.layers:
            encoder_output = layer(encoder_output, src_key_padding_mask=src_key_padding_mask)
            
        return encoder_output