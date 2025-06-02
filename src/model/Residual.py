import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor

class Residual(pl.LightningModule):
    """
    Residual connection with layer normalization as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    This implements the formula: LayerNorm(x + Sublayer(x)), where Sublayer is any layer function.
    The residual connections help with training deep networks by allowing gradients to flow
    through the network more easily. Layer normalization stabilizes the training process.
    
    In the Transformer architecture, each sublayer (self-attention or feed-forward network)
    in the encoder and decoder is wrapped with this residual connection and layer normalization.
    """
    def __init__(self, sublayer:nn.Module, dimension:int, dropout:float=0.1):
        """
        Initialize a Residual instance.
        
        Args:
            sublayer: The layer to be wrapped with residual connection and normalization
            dimension: The dimension of the input for layer normalization
            dropout: Dropout rate applied after the sublayer (default: 0.1)
        """
        super().__init__()
        self.sublayer = sublayer  # The layer to apply (attention or feed-forward)
        self.norm = nn.LayerNorm(dimension)  # Layer normalization
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
    
    def forward(self, *tensors:Tensor) -> Tensor:
        """
        Apply sublayer with residual connection and layer normalization.
        
        The implementation follows the formula: LayerNorm(x + Dropout(Sublayer(x)))
        where x is the last tensor in the input tensors.
        
        Args:
            *tensors: Input tensors to the sublayer. The last tensor is used for the residual connection.
            
        Returns:
            Output tensor after applying sublayer, dropout, residual connection, and layer normalization
        """
        # Apply sublayer to all input tensors, apply dropout, add residual connection, and normalize
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

