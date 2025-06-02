import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from model.SelfAttention import AttentionHead

class MultiHeadAttention(pl.LightningModule):
    """
    Multi-Head Attention module as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    Multi-head attention allows the model to jointly attend to information from different 
    representation subspaces at different positions. This is achieved by having multiple 
    attention heads operating in parallel, each with its own learned projection matrices.
    
    The outputs of these attention heads are concatenated and linearly transformed to 
    produce the final output.
    """
    def __init__(self, num_heads, dim_embedding, dim_k, dim_v):
        """
        Initialize a MultiHeadAttention instance.
        
        Args:
            num_heads: Number of attention heads
            dim_embedding: Dimension of the input embeddings
            dim_k: Dimension of keys for each attention head
            dim_v: Dimension of values for each attention head
        """
        super().__init__()
        
        # Create multiple attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(dim_embedding, dim_k, dim_v) for _ in range(num_heads)]
        )
        
        # Linear projection to combine outputs from all heads
        self.linear = nn.Linear(num_heads * dim_v, dim_embedding)
    
    def forward(self, query, key, value, mask=None) -> Tensor:
        """
        Compute multi-head attention.
        
        The process involves:
        1. Applying each attention head to the inputs
        2. Concatenating the results from all heads
        3. Applying a final linear transformation
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, dim_embedding]
            key: Key tensor of shape [batch_size, seq_len_k, dim_embedding]
            value: Value tensor of shape [batch_size, seq_len_v, dim_embedding]
            mask: Optional mask tensor to prevent attention to certain positions
                  Shape: [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim_embedding]
        """
        # Apply each attention head and concatenate results
        # Each head projects to dim_v, so concatenating gives num_heads * dim_v
        head_outputs = [head(query, key, value, mask=mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Apply final linear projection
        return self.linear(concatenated)

        
    
    

