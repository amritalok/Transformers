import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from utils.Helper import feed_forward
from model.MultiHeadAttention import MultiHeadAttention
from model.Residual import Residual
from typing import Optional

class EncoderLayer(pl.LightningModule):
    """
    EncoderLayer class implements a single layer of the Transformer encoder.
    
    As described in 'Attention is All You Need' (Vaswani et al., 2017), each encoder layer consists of:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    Both sublayers have residual connections and are followed by layer normalization.
    
    The encoder processes the input sequence and produces representations that are used by the decoder.
    """
    def __init__(
        self, 
        dim_embedding: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize an EncoderLayer instance.
        
        Args:
            dim_embedding: Dimension of the input embeddings (default: 512)
            num_heads: Number of attention heads (default: 6)
            dim_feedforward: Dimension of the feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Calculate dimension for each attention head
        dim_k = dim_v = dim_embedding // num_heads
        
        # Multi-head attention with residual connection and layer normalization
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )
        
        # Position-wise feed-forward network with residual connection and layer normalization
        self.feed_forward = Residual(
            feed_forward(dim_embedding, dim_feedforward),
            dimension = dim_embedding,
            dropout = dropout
        )
    
    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Process input through the encoder layer.
        
        The forward pass consists of:
        1. Self-attention where query, key, and value all come from the same source
        2. Feed-forward network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim_embedding]
            src_padding_mask: Optional mask for padding tokens in the source sequence
                              Shape: [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Processed tensor of the same shape as input
        """
        # Self-attention mechanism
        # self.attention is Residual(MHA_self_encoder)
        # Pass x for residual, then x,x,x for MHA's q,k,v, then mask.
        attention_output = self.attention(
            x,  # Input 'x' for the Residual connection
            # Arguments for the MultiHeadAttention sublayer:
            x,  # query
            x,  # key
            x,  # value
            attn_mask=None, # Encoder self-attention typically doesn't have a structural/causal mask
            key_padding_mask=src_key_padding_mask # Pass the source key padding mask
        )
        
        # Position-wise feed-forward network
        # self.feed_forward is Residual(FeedForwardNetwork)
        # Pass attention_output for residual, then attention_output as input to FFN.
        ff_output = self.feed_forward(attention_output, attention_output)
        return ff_output
