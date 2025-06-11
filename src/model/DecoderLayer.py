import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from utils.Helper import feed_forward, create_look_ahead_mask
from model.MultiHeadAttention import MultiHeadAttention
from model.Residual import Residual
from typing import Optional

class DecoderLayer(pl.LightningModule):
    """
    DecoderLayer class implements a single layer of the Transformer decoder.
    
    As described in 'Attention is All You Need' (Vaswani et al., 2017), each decoder layer consists of:
    1. Masked multi-head self-attention mechanism (prevents attending to future positions)
    2. Multi-head cross-attention mechanism (attends to encoder outputs)
    3. Position-wise feed-forward network
    
    All three sublayers have residual connections and are followed by layer normalization.
    
    The decoder processes the target sequence and encoder outputs to generate predictions.
    """
    def __init__(
        self,
        dim_embedding: int = 512,
        num_heads: int = 6,
        dim_feedfordward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize a DecoderLayer instance.
        
        Args:
            dim_embedding: Dimension of the input embeddings (default: 512)
            num_heads: Number of attention heads (default: 6)
            dim_feedfordward: Dimension of the feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Calculate dimension for each attention head
        dim_k = dim_v = dim_embedding // num_heads
        
        # First attention block: masked self-attention
        # This prevents the decoder from attending to future positions in the target sequence
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )

        # Second attention block: cross-attention between decoder and encoder
        # This allows the decoder to attend to all positions in the encoder output
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_embedding, dim_k, dim_v),
            dimension = dim_embedding,
            dropout = dropout
        )

        # Position-wise feed-forward network
        self.feed_forward = Residual(
            feed_forward(dim_embedding, dim_feedfordward),
            dimension = dim_embedding,
            dropout = dropout
        )

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_self_attn_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None
            ) -> Tensor:
        """
        Process input through the decoder layer.
        
        The forward pass consists of:
        1. Masked self-attention (prevents attending to future positions)
        2. Cross-attention between decoder and encoder outputs
        3. Feed-forward network
        
        Args:
            x: Target sequence tensor of shape [batch_size, seq_len, dim_embedding]
            
        Returns:
            Processed tensor of shape [batch_size, seq_len, dim_embedding]
        """
        # tgt: target sequence embeddings [batch_size, tgt_seq_len, dim_embedding]
        # memory: encoder output [batch_size, src_seq_len, dim_embedding]
        # tgt_self_attn_mask: causal mask for self-attention [tgt_seq_len, tgt_seq_len], True means ignore.
        # memory_key_padding_mask: padding mask for memory [batch_size, src_seq_len], True means ignore padding.
        # tgt_key_padding_mask: padding mask for target [batch_size, tgt_seq_len], True means ignore padding.

        # Step 1: Masked self-attention on the target sequence
        # Query, Key, Value are all from `tgt`.
        # MultiHeadAttention expects `attn_mask` (for causal/structural) and `key_padding_mask` (for padding).
        self_attn_out = self.attention_1(
            tgt,  # Input 'x' for the Residual connection
            # Arguments for the MultiHeadAttention sublayer:
            tgt,  # query
            tgt,  # key
            tgt,  # value
            attn_mask=tgt_self_attn_mask,      # Causal mask (e.g., [tgt_seq_len, tgt_seq_len])
            key_padding_mask=tgt_key_padding_mask  # Padding in target sequence (e.g., [batch_size, tgt_seq_len])
        )
        
        # Step 2: Cross-attention with encoder output (memory)
        # Query comes from self_attn_out (output of first sublayer).
        # Key and Value come from `memory` (encoder output).
        # MultiHeadAttention expects `attn_mask` (causal/structural, usually None for cross-attention) 
        # and `key_padding_mask` (for padding in memory).
        cross_attn_out = self.attention_2(
            self_attn_out,  # Input 'x' for the Residual connection
            # Arguments for the MultiHeadAttention sublayer:
            self_attn_out,  # query
            memory,         # key
            memory,         # value
            attn_mask=None, # Typically no structural/causal mask for cross-attention
            key_padding_mask=memory_key_padding_mask # Padding in encoder memory (e.g., [batch_size, src_seq_len])
        )
        
        # Step 3: Feed-forward network
        ff_output = self.feed_forward(cross_attn_out, cross_attn_out)
        return ff_output
