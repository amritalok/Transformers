import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from utils.Helper import feed_forward, create_look_ahead_mask
from model.MultiHeadAttention import MultiHeadAttention
from model.Residual import Residual

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

    def forward(self, x:Tensor, encoder_output:Tensor) -> Tensor:
        """
        Process input through the decoder layer.
        
        The forward pass consists of:
        1. Masked self-attention (prevents attending to future positions)
        2. Cross-attention between decoder and encoder outputs
        3. Feed-forward network
        
        Args:
            x: Target sequence tensor of shape [batch_size, seq_len, dim_embedding]
            encoder_output: Encoder output tensor of shape [batch_size, src_seq_len, dim_embedding]
            
        Returns:
            Processed tensor of shape [batch_size, seq_len, dim_embedding]
        """
        # Create look-ahead mask for decoder self-attention
        seq_len = x.size(1)
        look_ahead_mask = create_look_ahead_mask(seq_len, device=x.device)
        # Expand mask to match batch size
        batch_size = x.size(0)
        look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Step 1: Apply masked self-attention to target sequence
        # This prevents positions from attending to subsequent positions
        # In self-attention, query, key, and value all come from the same source (x)
        self_attention_output = self.attention_1(x, x, x, mask=look_ahead_mask)
        
        # Step 2: Apply cross-attention between decoder and encoder outputs
        # Query comes from decoder (self_attention_output), Key and Value come from encoder (encoder_output)
        cross_attention_output = self.attention_2(self_attention_output, encoder_output, encoder_output)
        
        # Step 3: Apply feed-forward network
        return self.feed_forward(cross_attention_output)


