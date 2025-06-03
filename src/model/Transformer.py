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
        # Get sequence lengths and batch size
        batch_size = src_tokens.size(0)
        src_seq_len = src_tokens.size(1)
        tgt_seq_len = tgt_tokens.size(1)
        
        # Create padding mask for source sequence (1 for tokens, 0 for padding)
        # This will be used to mask out padding tokens in attention
        src_padding_mask = (src_tokens != self.pad_idx).float()  # Shape: [batch_size, src_seq_len]
        
        # Convert token IDs to embeddings and scale
        src_embeddings = self.src_embedding(src_tokens) * self.embedding_scale
        tgt_embeddings = self.tgt_embedding(tgt_tokens) * self.embedding_scale
        
        # Add positional encoding, handling possible sequence length differences
        max_pos_len = self.positional_encoding.size(1)
        
        # Ensure we don't exceed the pre-computed positional encoding length
        src_seq_len_capped = min(src_seq_len, max_pos_len)
        tgt_seq_len_capped = min(tgt_seq_len, max_pos_len)
        
        # If source sequence is too long, truncate it for positional encoding
        if src_seq_len > max_pos_len:
            print(f"Warning: Source sequence length {src_seq_len} exceeds maximum positional encoding length {max_pos_len}")
            src_embeddings = src_embeddings[:, :max_pos_len, :]
            src_padding_mask = src_padding_mask[:, :max_pos_len]
            src_seq_len = max_pos_len
        
        # If target sequence is too long, truncate it for positional encoding
        if tgt_seq_len > max_pos_len:
            print(f"Warning: Target sequence length {tgt_seq_len} exceeds maximum positional encoding length {max_pos_len}")
            tgt_embeddings = tgt_embeddings[:, :max_pos_len, :]
            tgt_tokens = tgt_tokens[:, :max_pos_len]
            tgt_seq_len = max_pos_len
        
        # Add positional encoding
        src_embeddings = src_embeddings + self.positional_encoding[:, :src_seq_len, :]
        tgt_embeddings = tgt_embeddings + self.positional_encoding[:, :tgt_seq_len, :]
        
        # Apply dropout
        src_embeddings = self.dropout(src_embeddings)
        tgt_embeddings = self.dropout(tgt_embeddings)
        
        # First encode the source sequence
        encoder_output = self.encoder(src_embeddings)
        
        # Then decode using the encoded source and target embeddings
        # Pass the source padding mask to handle different sequence lengths
        return self.decoder(tgt_embeddings, encoder_output, src_padding_mask)
    
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

    @torch.no_grad()
    def generate(self, src_tokens: torch.Tensor, max_length: int = 50, decoder_start_token_id: int = None, num_beams: int = 1, early_stopping: bool = True, length_penalty: float = 1.0):
        # num_beams, early_stopping, length_penalty are common HuggingFace generate args,
        # but this basic greedy version won't use them. They are included for signature compatibility.
        self.eval() 
        device = src_tokens.device
        batch_size = src_tokens.size(0)

        if decoder_start_token_id is None:
            # Fallback, though it should be provided by the caller.
            # MarianTokenizer's BOS is often its PAD token or a language code.
            # Using self.pad_idx as a guess if nothing else is known.
            effective_bos_token_id = self.pad_idx 
            print(f"Warning: decoder_start_token_id not provided to generate, using self.pad_idx ({self.pad_idx}) as fallback BOS.")
        else:
            effective_bos_token_id = decoder_start_token_id

        # 1. Encode the source tokens
        # (Reusing logic from the forward pass for embedding and positional encoding)
        src_embeddings = self.src_embedding(src_tokens) * self.embedding_scale
        # Determine sequence length for positional encoding, capped by PE buffer size
        src_seq_len_for_pe = min(src_tokens.size(1), self.positional_encoding.size(1))
        
        # Adjust src_embeddings if its sequence length is greater than positional_encoding's max length
        current_src_seq_len = src_embeddings.size(1)
        if current_src_seq_len > self.positional_encoding.size(1):
            src_embeddings_adjusted = src_embeddings[:, :self.positional_encoding.size(1), :]
        else:
            src_embeddings_adjusted = src_embeddings

        src_embeddings_final = src_embeddings_adjusted + self.positional_encoding[:, :src_embeddings_adjusted.size(1), :] # Use adjusted length for PE slicing
        src_embeddings_final = self.dropout(src_embeddings_final)
        encoder_output = self.encoder(src_embeddings_final)

        # 2. Initialize decoder input with BOS token
        tgt_tokens = torch.full((batch_size, 1), effective_bos_token_id, dtype=torch.long, device=device)

        # 3. Autoregressive decoding loop
        for _ in range(max_length - 1): 
            tgt_embeddings = self.tgt_embedding(tgt_tokens) * self.embedding_scale
            
            # Determine sequence length for positional encoding, capped by PE buffer size
            current_tgt_seq_len = tgt_embeddings.size(1)
            if current_tgt_seq_len > self.positional_encoding.size(1):
                tgt_embeddings_adjusted = tgt_embeddings[:, :self.positional_encoding.size(1), :]
            else:
                tgt_embeddings_adjusted = tgt_embeddings
                
            tgt_embeddings_final = tgt_embeddings_adjusted + self.positional_encoding[:, :tgt_embeddings_adjusted.size(1), :] # Use adjusted length for PE slicing
            tgt_embeddings_final = self.dropout(tgt_embeddings_final)
            
            # Create source padding mask for the decoder's cross-attention
            src_padding_mask = (src_tokens != self.pad_idx).float()

            decoder_output_logits = self.decoder(tgt_embeddings_final, encoder_output, src_padding_mask=src_padding_mask)
            
            next_token_logits = decoder_output_logits[:, -1, :] 
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            tgt_tokens = torch.cat((tgt_tokens, next_token), dim=1)

            # Optional: Stop if an EOS token is generated. Requires EOS token ID.
            # eos_token_id = getattr(self, 'eos_token_id', None) # Define eos_token_id if needed
            # if eos_token_id is not None and (next_token == eos_token_id).all():
            #     break
        
        return tgt_tokens

