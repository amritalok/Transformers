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
    
    def forward(self, x_for_residual: Tensor, *args_for_sublayer: Tensor, **kwargs_for_sublayer) -> Tensor:
        """
        Apply sublayer with residual connection and layer normalization.
        
        The implementation follows the formula: LayerNorm(x + Dropout(Sublayer(x)))
        
        Args:
            x_for_residual: The input tensor for the residual connection (the 'x' in "x + sublayer()").
            *args_for_sublayer: Positional arguments to be passed to the sublayer.
            **kwargs_for_sublayer: Keyword arguments to be passed to the sublayer (e.g., mask).
            
        Returns:
            Output tensor after applying sublayer, dropout, residual connection, and layer normalization
        """
        try:
            # Apply sublayer with its specific arguments
            sublayer_output = self.sublayer(*args_for_sublayer, **kwargs_for_sublayer)
            
            # Apply dropout, add residual connection (with x_for_residual), and normalize
            # Ensure shapes match for the addition if necessary, though usually sublayer output matches query/input.
            if x_for_residual.shape != sublayer_output.shape:
                # This might indicate an issue elsewhere if shapes are not compatible for addition.
                # For attention, output shape matches query shape. For FFN, output shape matches input shape.
                # Handling mismatches here can be complex; ideally, they should match.
                # For now, we'll assume they match or the sublayer handles it.
                # A more robust solution might involve padding/truncating or specific error handling.
                pass # Or add specific error logging/handling if shapes can mismatch

            return self.norm(x_for_residual + self.dropout(sublayer_output))
        except Exception as e:
            # For debugging during development
            print(f"Error in Residual forward: {e}")
            print(f"  - x_for_residual shape: {x_for_residual.shape}")
            print(f"  - args_for_sublayer shapes: {[t.shape for t in args_for_sublayer]}")
            print(f"  - kwargs_for_sublayer: {kwargs_for_sublayer}")
            raise

