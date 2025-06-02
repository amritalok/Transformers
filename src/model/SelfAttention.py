import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from utils.Helper import scaled_dot_product_attention

class AttentionHead(pl.LightningModule):
    """
    Single Attention Head module as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    Each attention head performs the scaled dot-product attention mechanism on a different
    subspace of the input representations. This allows the model to focus on different aspects
    of the input sequence.
    
    The attention mechanism computes a weighted sum of values (V), where the weights are
    determined by the compatibility between the query (Q) and key (K) representations.
    """
    def __init__(self, dim_model:int, dim_key:int, dim_value:int):
        """
        Initialize an AttentionHead instance.
        
        Args:
            dim_model: Dimension of the input embeddings
            dim_key: Dimension to project keys and queries to
            dim_value: Dimension to project values to
        """
        super().__init__()
        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(dim_model, dim_key, bias=False)
        self.key_proj = nn.Linear(dim_model, dim_key, bias=False)
        self.value_proj = nn.Linear(dim_model, dim_value, bias=False)
    
    def forward(self, query:Tensor, key:Tensor, value:Tensor, mask=None) -> Tensor:
        """
        Compute scaled dot-product attention for a single head.
        
        The process involves:
        1. Projecting queries, keys, and values to their respective spaces
        2. Computing attention scores between queries and keys
        3. Applying mask (if provided) to prevent attention to certain positions
        4. Computing weighted sum of values based on attention scores
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, dim_model]
            key: Key tensor of shape [batch_size, seq_len_k, dim_model]
            value: Value tensor of shape [batch_size, seq_len_v, dim_model]
            mask: Optional mask tensor to prevent attention to certain positions
                  Shape: [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim_value]
        """
        # Project queries, keys, and values to their respective spaces
        query_projected = self.query_proj(query)
        key_projected = self.key_proj(key)
        value_projected = self.value_proj(value)
        
        # Compute scaled dot-product attention
        return scaled_dot_product_attention(query_projected, key_projected, value_projected, mask=mask)
    


