import math
import torch
from torch import nn
import pytorch_lightning as pl
from torch import Tensor
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(pl.LightningModule):
    """
    Transformer model as described in 'Attention is All You Need' (Vaswani et al., 2017).
    
    The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms,
    without using recurrence or convolution. It consists of an encoder and a decoder, each composed
    of a stack of identical layers.
    
    The encoder maps an input sequence to a sequence of continuous representations,
    which the decoder then uses to generate an output sequence one element at a time.
    
    This implementation uses PyTorch Lightning for training and organization.
    """
    def __init__(
        self,
        src_vocab_size: int,  # Size of source vocabulary
        tgt_vocab_size: int,  # Size of target vocabulary
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_embedding: int = 512,
        num_heads: int = 8,
        dim_feedfordward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        pad_idx: int = 0,  # Padding index for loss calculation
        max_seq_length: int = 5000  # Maximum sequence length for positional encoding
    ):
        """
        Initialize a Transformer instance.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            num_encoder_layers: Number of layers in the encoder (default: 6)
            num_decoder_layers: Number of layers in the decoder (default: 6)
            dim_embedding: Dimension of the embeddings (default: 512)
            num_heads: Number of attention heads (default: 8)
            dim_feedfordward: Dimension of the feedforward network (default: 2048)
            dropout: Dropout rate (default: 0.1)
            activation: Activation function for the feedforward network (default: ReLU)
            pad_idx: Index used for padding tokens (default: 0)
            max_seq_length: Maximum sequence length for positional encoding (default: 5000)
        """
        super().__init__()
        
        # Token embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, dim_embedding, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim_embedding, padding_idx=pad_idx)
        
        # Scale embeddings as per the paper (multiply by sqrt(d_model))
        self.embedding_scale = math.sqrt(dim_embedding)
        
        # Positional encoding
        self.register_buffer(
            "positional_encoding", 
            self._create_positional_encoding(max_seq_length, dim_embedding)
        )
        
        # Dropout after embedding and positional encoding
        self.dropout = nn.Dropout(dropout)
        
        # Encoder stack - processes the input sequence
        self.encoder = Encoder(
            num_layers = num_encoder_layers,
            dim_embedding = dim_embedding,
            num_heads = num_heads,
            dim_feedfordward = dim_feedfordward,
            dropout = dropout
        )

        # Decoder stack - generates the output sequence
        self.decoder = Decoder(
            num_layers = num_decoder_layers,
            dim_embedding = dim_embedding,
            num_heads = num_heads,
            dim_feedfordward = dim_feedfordward,
            dropout = dropout,
            vocab_size = tgt_vocab_size
        )
        
        # Store padding index for loss calculation
        self.pad_idx = pad_idx
        
        # Loss function that ignores padding tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    def _create_positional_encoding(self, max_seq_length, dim_embedding):
        """
        Create positional encodings for the inputs.
        
        Args:
            max_seq_length: Maximum sequence length
            dim_embedding: Dimension of the embeddings
            
        Returns:
            Positional encoding tensor of shape [1, max_seq_length, dim_embedding]
        """
        # Create position indices
        positions = torch.arange(max_seq_length).unsqueeze(1).float()
        # Create dimension indices
        dimensions = torch.arange(0, dim_embedding, 2).float()
        
        # Create the angle rates for the positional encoding formula
        angle_rates = 1 / torch.pow(10000, (dimensions / dim_embedding))
        
        # Create the positional encoding
        pos_encoding = torch.zeros(max_seq_length, dim_embedding)
        pos_encoding[:, 0::2] = torch.sin(positions * angle_rates)  # Even dimensions
        pos_encoding[:, 1::2] = torch.cos(positions * angle_rates)  # Odd dimensions
        
        # Add batch dimension and return
        return pos_encoding.unsqueeze(0)

    def forward(self, src_tokens:Tensor, tgt_tokens:Tensor) -> Tensor:
        """
        Process source and target sequences through the Transformer.
        
        Args:
            src_tokens: Source token IDs of shape [batch_size, src_seq_len]
            tgt_tokens: Target token IDs of shape [batch_size, tgt_seq_len]
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, tgt_vocab_size] with probabilities
            after softmax activation
        """
        # Get sequence lengths for positional encoding
        src_seq_len = src_tokens.size(1)
        tgt_seq_len = tgt_tokens.size(1)
        
        # Convert token IDs to embeddings and scale
        src_embeddings = self.src_embedding(src_tokens) * self.embedding_scale
        tgt_embeddings = self.tgt_embedding(tgt_tokens) * self.embedding_scale
        
        # Add positional encoding
        src_embeddings = src_embeddings + self.positional_encoding[:, :src_seq_len, :]
        tgt_embeddings = tgt_embeddings + self.positional_encoding[:, :tgt_seq_len, :]
        
        # Apply dropout
        src_embeddings = self.dropout(src_embeddings)
        tgt_embeddings = self.dropout(tgt_embeddings)
        
        # First encode the source sequence
        encoder_output = self.encoder(src_embeddings)
        
        # Then decode using the encoded source and target embeddings
        return self.decoder(tgt_embeddings, encoder_output)
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (input, target) tensors
            batch_idx: Index of the current batch
            
        Returns:
            Calculated loss value
        """
        src_tokens, tgt_tokens_input, tgt_tokens_target = batch
        
        # Forward pass through the model
        # tgt_tokens_input is the target sequence shifted right (for teacher forcing)
        # tgt_tokens_target is the actual target sequence to predict
        outputs = self(src_tokens, tgt_tokens_input)
        
        # Reshape predictions and targets for loss calculation
        # from [batch_size, seq_len, vocab_size] to [batch_size * seq_len, vocab_size]
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens_target.view(-1))
        
        # Log the training loss
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate accuracy (for monitoring)
        with torch.no_grad():
            predictions = outputs.argmax(dim=-1)
            mask = (tgt_tokens_target != self.pad_idx)
            accuracy = (predictions == tgt_tokens_target)[mask].float().mean()
            self.log('train_accuracy', accuracy, prog_bar=True)
            
        return loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
            Adam optimizer with learning rate 1e-4
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
