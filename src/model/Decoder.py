import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from model.DecoderLayer import DecoderLayer
from typing import Optional

class Decoder(pl.LightningModule):
    """
    Decoder module of the Transformer architecture as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    The Decoder consists of a stack of identical layers, each containing:
    1. Masked multi-head self-attention mechanism
    2. Multi-head cross-attention mechanism to encoder outputs
    3. Position-wise feed-forward network
    
    The decoder takes the target sequence (shifted right) and the encoder output as input
    and generates predictions for the next tokens in the sequence.
    """
    def __init__(
        self, 
        num_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 8,
        dim_feedfordward: int = 2048,
        dropout: int = 0.1,
        vocab_size: int = None  # Vocabulary size for output projection
    ):
        """
        Initialize a Decoder instance.
        
        Args:
            num_layers: Number of decoder layers (default: 6)
            dim_embedding: Dimension of the input embeddings (default: 512)
            num_heads: Number of attention heads (default: 8)
            dim_feedfordward: Dimension of the feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
            vocab_size: Size of the vocabulary for output projection (default: None)
                        If None, uses dim_embedding as the output dimension
        """
        super().__init__()
        
        # Stack of identical decoder layers
        self.layers = nn.ModuleList([ 
            DecoderLayer(dim_embedding, num_heads, dim_feedfordward, dropout) for _ in range(num_layers)
        ])
        
        # Final linear projection to vocabulary size (or embedding dimension if vocab_size is None)
        self.output_dim = vocab_size if vocab_size is not None else dim_embedding
        self.linear = nn.Linear(dim_embedding, self.output_dim)
    

    def forward(self, 
                tgt_embed: Tensor, 
                memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None
            ) -> Tensor:
        """
        Process target sequence and encoder output through the decoder.
        
        The forward pass consists of:
        1. Processing through each decoder layer
        2. Final linear projection and softmax
        
        Args:
            x: Target sequence tensor of shape [batch_size, seq_len, dim_embedding] with positional encoding
            encoder_output: Encoder output tensor of shape [batch_size, src_seq_len, dim_embedding]
            src_padding_mask: Optional mask for padding tokens in the source sequence,
                             with shape [batch_size, src_seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim] with probabilities
            after softmax activation
        """
        # Process through each decoder layer
        # x is the target embeddings (tgt_embed)
        # memory is the encoder_output
        # tgt_mask is the causal mask for self-attention
        # memory_key_padding_mask is the padding mask for the encoder output (source padding)
        # tgt_key_padding_mask is the padding mask for the target sequence itself
        current_input = tgt_embed
        for layer in self.layers:
            current_input = layer(
                current_input,  # target input to the layer
                memory,  # encoder output (memory)
                tgt_self_attn_mask=tgt_mask, # causal mask for self-attention
                tgt_key_padding_mask=tgt_key_padding_mask, # target padding for self-attention
                memory_key_padding_mask=memory_key_padding_mask # source padding for cross-attention
            )
        
        # Final linear projection (decoder_output is now current_input)
        logits = self.linear(current_input)
        return logits # Return raw logits
