import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import re
import math
from typing import Dict, Any, Optional


class Config(dict):
    """
    A dictionary-like class that allows attribute-style access to its keys.
    
    Example:
        config = Config({"a": 1, "b": 2})
        print(config.a)  # Output: 1
        config.c = 3
        print(config["c"])  # Output: 3
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
        
    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.items())
        
    def __repr__(self):
        return f"Config({super().__repr__()})"

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    """
    Compute scaled dot-product attention as described in 'Attention is All You Need'.
    This implementation handles attention between sequences of different lengths.
    
    The attention mechanism computes a weighted sum of values, where the weights are
    determined by the compatibility between the query and key representations.
    
    Args:
        query: Query tensor of shape [batch_size, seq_len_q, dim_k]
        key: Key tensor of shape [batch_size, seq_len_k, dim_k]
        value: Value tensor of shape [batch_size, seq_len_k, dim_v]
        mask: Optional mask tensor of shape [batch_size, seq_len_q, seq_len_k]
              or [seq_len_q, seq_len_k]
              Values of 0 in the mask prevent attention to the corresponding positions.
        
    Returns:
        Output tensor of shape [batch_size, seq_len_q, dim_v]
    """
    # Get dimensions
    batch_size = query.size(0)
    seq_len_q = query.size(1)
    seq_len_k = key.size(1)
    dim_k = query.size(-1)
    dim_v = value.size(-1)
    
    # Print tensor dimensions for debugging
    # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
    
    # Compute attention scores: batch_matmul(Q, K^T) / sqrt(d_k)
    # Shape: [batch_size, seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    
    # Apply attention mask (if provided)
    # attn_mask should be boolean, where True indicates a position that should NOT be attended to.
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            # Assuming 0 means ignore if not boolean (for backward compatibility with old float masks if any)
            # However, boolean masks are strongly preferred.
            print(f"Warning: attn_mask in scaled_dot_product_attention is not boolean (dtype: {attn_mask.dtype}). Assuming 0 means ignore.")
            scores = scores.masked_fill(attn_mask == 0, -float('inf'))
        else:
            scores = scores.masked_fill(attn_mask, -float('inf'))
        
        # Ensure the mask has the correct shape for broadcasting or direct application.
        # Expected shapes for attn_mask: 
        #   - [batch_size, seq_len_q, seq_len_k] (for batch-specific padding masks combined with causal)
        #   - [seq_len_q, seq_len_k] (for causal mask, will broadcast across batch)
        #   - [batch_size, 1, seq_len_k] (for key_padding_mask, will broadcast across query_len)
        # The `masked_fill` handles broadcasting if dimensions are compatible (e.g., [L,S] mask for [B,H,L,S] scores in nn.MHA)
        # Here, scores are [B, L, S], so mask should be [B,L,S] or [L,S] or [B,1,S] or [B,L,1]

    # Apply softmax to get attention weights
    # Shape: [batch_size, seq_len_q, seq_len_k]
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    # Shape: [batch_size, seq_len_q, dim_v]
    output = torch.matmul(attention_weights, value)
    
    return output


def create_look_ahead_mask(size: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Create a look-ahead mask to prevent positions from attending to subsequent positions.
    
    This is used in the decoder to ensure that predictions for position i can only
    depend on known outputs at positions less than i.
    
    Args:
        size: Size of the square mask
        device: Device to create the mask on
        
    Returns:
        Mask tensor of shape [size, size] where 1 means attend, 0 means don't attend
    """
    # Create a square matrix with True in the upper triangle (positions to be masked).
    # True means the position should NOT be attended to.
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask


def feed_forward(dim_embedding: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    """
    Create a position-wise feed-forward network as described in 'Attention is All You Need'.
    
    The feed-forward network consists of two linear transformations with a ReLU activation
    in between: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Args:
        dim_embedding: Input and output dimension (default: 512)
        dim_feedforward: Inner dimension of the FFN (default: 2048)
        
    Returns:
        A PyTorch module implementing the feed-forward network
    """
    return nn.Sequential(
        nn.Linear(dim_embedding, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_embedding)
    )


def preprocess(sentence: str) -> str:
    """
    Preprocess a text sentence for NLP tasks.
    
    This function performs the following operations:
    1. Strips leading and trailing whitespace
    2. Replaces special characters with spaces
    3. Normalizes whitespace to single spaces
    4. Normalizes repeated punctuation (!, ?, ,)
    5. Converts text to lowercase
    
    Args:
        sentence: The input text to preprocess
        
    Returns:
        Preprocessed text string
    """
    # Convert to string and strip whitespace
    sentence = str(sentence).strip()
    
    # Replace special characters with spaces
    sentence = re.sub(
        r"[\*\"""\n\\…\+\-\/\=\(\)'•:\[\]\|'\!;]", " ", sentence)
    
    # Normalize whitespace
    sentence = re.sub(r"[ ]+", " ", sentence)
    
    # Normalize repeated punctuation
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    
    # Convert to lowercase
    sentence = sentence.lower()
    
    return sentence
