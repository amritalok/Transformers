#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer training script using Hugging Face's Trainer API.

This script uses the custom Transformer implementation from the 'model' directory
while leveraging Hugging Face's datasets and Trainer API for efficient training.
Optimized for M3 Mac Pro with Metal Performance Shaders (MPS) acceleration.
"""

import os
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

# Hugging Face imports
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import (
    MarianTokenizer,  # Specialized tokenizer for translation tasks
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    TrainerCallback  # For custom callbacks
)

# Import custom Transformer model
from model.Transformer import Transformer

# Custom callback for saving last checkpoint
class SaveLastCheckpointCallback(TrainerCallback):
    """Callback to save the model and tokenizer at the end of training."""
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Save model and tokenizer when training ends."""
        callback_logger = logging.getLogger(f"{__name__}.SaveLastCheckpointCallback")
        callback_logger.info("SaveLastCheckpointCallback.on_train_end called.")
        if model is not None and tokenizer is not None:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint-last")
            callback_logger.info(f"Attempting to save final model checkpoint to {checkpoint_dir}")
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                callback_logger.info(f"Ensured directory exists: {checkpoint_dir}")
                
                model.save_pretrained(checkpoint_dir)
                callback_logger.info(f"Model saved to {checkpoint_dir}")
                
                tokenizer.save_pretrained(checkpoint_dir)
                callback_logger.info(f"Tokenizer saved to {checkpoint_dir}")
                
                callback_logger.info(f"Final checkpoint (model and tokenizer) saved successfully to {checkpoint_dir}")
            except Exception as e:
                callback_logger.error(f"Error during SaveLastCheckpointCallback in on_train_end: {e}", exc_info=True)
        else:
            callback_logger.warning("SaveLastCheckpointCallback.on_train_end: Model or Tokenizer is None. Cannot save checkpoint-last.")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the custom Transformer model configuration.
    """
    dim_embedding: int = field(
        default=512,
        metadata={"help": "Dimension of embeddings in the Transformer model"}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the Transformer model"}
    )
    num_encoder_layers: int = field(
        default=6,
        metadata={"help": "Number of layers in the Transformer encoder"}
    )
    num_decoder_layers: int = field(
        default=6, 
        metadata={"help": "Number of layers in the Transformer decoder"}
    )
    dim_feedforward: int = field(
        default=2048,
        metadata={"help": "Dimension of the feedforward network in Transformer layers"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate in the Transformer model"}
    )
    tokenizer_name: str = field(
        default="Helsinki-NLP/opus-mt-fr-en",
        metadata={"help": "Tokenizer to use for preprocessing, default is MarianTokenizer for fr-en"}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset_name: str = field(
        default="wmt14",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_fraction: float = field(
        default=0.25,
        metadata={"help": "Fraction of the dataset to use for each split (between 0 and 1). Default is 0.25 (25%)."}
    )
    dataset_config_name: Optional[str] = field(
        default="fr-en",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    source_lang: str = field(default="fr", metadata={"help": "Source language id for translation."})
    target_lang: str = field(default="en", metadata={"help": "Target language id for translation."})
    max_source_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    pad_token: str = field(
        default="<pad>",
        metadata={"help": "Padding token"}
    )
    bos_token: str = field(
        default="<bos>",
        metadata={"help": "Beginning of sequence token"}
    )
    eos_token: str = field(
        default="<eos>",
        metadata={"help": "End of sequence token"}
    )
    unk_token: str = field(
        default="<unk>",
        metadata={"help": "Unknown token"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training and evaluation"}
    )


# Custom data collator to handle teacher forcing in our Transformer model
class CustomDataCollator:
    def __init__(self, tokenizer, pad_token_id, device):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.device = device
    
    def __call__(self, features):
        # Ensure features contain tensors (convert lists to tensors if needed)
        # This is necessary because the dataset might contain lists instead of tensors
        if isinstance(features[0]['input_ids'], list):
            # Convert lists to tensors
            input_ids = [torch.tensor(feature['input_ids']) for feature in features]
            attention_mask = [torch.tensor(feature['attention_mask']) for feature in features]
            labels = [torch.tensor(feature['labels']) for feature in features]
            
            # Pad sequences to the same length if necessary
            max_input_len = max(len(ids) for ids in input_ids)
            max_label_len = max(len(lbl) for lbl in labels)
            
            # Pad input_ids and attention_mask
            padded_input_ids = []
            padded_attention_mask = []
            for ids, mask in zip(input_ids, attention_mask):
                if len(ids) < max_input_len:
                    padding = torch.full((max_input_len - len(ids),), self.pad_token_id, dtype=ids.dtype)
                    ids = torch.cat([ids, padding])
                    mask_padding = torch.zeros(max_input_len - len(mask), dtype=mask.dtype)
                    mask = torch.cat([mask, mask_padding])
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
            
            # Pad labels
            padded_labels = []
            for lbl in labels:
                if len(lbl) < max_label_len:
                    padding = torch.full((max_label_len - len(lbl),), self.pad_token_id, dtype=lbl.dtype)
                    lbl = torch.cat([lbl, padding])
                padded_labels.append(lbl)
            
            # Stack tensors
            input_ids = torch.stack(padded_input_ids)
            attention_mask = torch.stack(padded_attention_mask)
            labels = torch.stack(padded_labels)
        else:
            # Features already contain tensors, just stack them
            input_ids = torch.stack([feature['input_ids'] for feature in features])
            attention_mask = torch.stack([feature['attention_mask'] for feature in features])
            labels = torch.stack([feature['labels'] for feature in features])
        
        # For teacher forcing, we need target_input (shifted right) and target_output
        # Target input is the labels but with the last token removed (we'll add BOS at the beginning)
        target_input = labels[:, :-1].clone()
        
        # Make sure we have a valid BOS token ID
        bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None else 2  # Default to 2 if not found
        
        # Add BOS token at the beginning
        bos_tensor = torch.full(
            (target_input.size(0), 1), 
            bos_token_id, 
            dtype=target_input.dtype
        )
        target_input = torch.cat([bos_tensor, target_input], dim=1)
        
        # Target output is the labels but with the first token removed
        target_output = labels[:, 1:].clone()
        
        # Handle padding (-100 is ignored in loss calculation)
        target_output = torch.where(target_output == self.pad_token_id, -100, target_output)
        
        # We'll move tensors to the appropriate device in the Trainer
        return {
            'src_tokens': input_ids,
            'tgt_tokens_input': target_input,
            'tgt_tokens_target': target_output,
            'attention_mask': attention_mask
        }


# Custom Trainer class to work with our Transformer model
class TransformerTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        # Add the argument to keep all columns in the dataset
        if 'args' in kwargs and not hasattr(kwargs['args'], 'remove_unused_columns'):
            kwargs['args'].remove_unused_columns = False
        super().__init__(*args, **kwargs)
    
    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model. This version assumes the CustomDataCollator
        has already prepared src_tokens, tgt_tokens_input, and tgt_tokens_target.
        Relies on super()._prepare_inputs() for device placement and AMP handling.
        Ensures 'tgt_tokens' (expected by model.forward) is mapped from 'tgt_tokens_input'.
        """
        prepared_inputs = super()._prepare_inputs(inputs)

        if 'tgt_tokens_input' in prepared_inputs and 'tgt_tokens' not in prepared_inputs:
            prepared_inputs['tgt_tokens'] = prepared_inputs['tgt_tokens_input']
        elif 'tgt_tokens' in prepared_inputs and 'tgt_tokens_input' not in prepared_inputs:
             prepared_inputs['tgt_tokens_input'] = prepared_inputs['tgt_tokens']
        
        return prepared_inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        src_tokens = inputs.get("src_tokens")
        tgt_tokens_input_for_model = inputs.get("tgt_tokens") 
        tgt_tokens_target = inputs.get("tgt_tokens_target")
        
        if src_tokens is None or tgt_tokens_input_for_model is None or tgt_tokens_target is None:
            logger = logging.getLogger(__name__)
            logger.error(f"compute_loss missing critical inputs. Available: {list(inputs.keys())}")
            if return_outputs:
                return torch.tensor(float('nan'), device=model.device), {}
            return torch.tensor(float('nan'), device=model.device)

        outputs = model(src_tokens, tgt_tokens_input_for_model)
        
        # outputs shape: [B, S_out, V], tgt_tokens_target shape: [B, S_tgt]
        # S_out (from decoder based on tgt_tokens_input) is max_label_len
        # S_tgt (from collator based on labels[:, 1:]) is max_label_len - 1
        # Slice outputs to match the target length for loss calculation.
        # We want to compare predictions for t_1...t_L-1 with actual t_1...t_L-1.
        # The 0-th output logit corresponds to predicting the token at target index 0 (which is labels[1]).
        # The (L-2)-th output logit corresponds to predicting token at target index L-2 (which is labels[L-1]).
        # So, we need outputs up to sequence length L-1 (i.e., outputs[:, :tgt_tokens_target.shape[1], :]).
        sliced_outputs = outputs[:, :tgt_tokens_target.shape[1], :].contiguous()
        
        reshaped_outputs = sliced_outputs.reshape(-1, sliced_outputs.shape[-1])
        reshaped_targets = tgt_tokens_target.reshape(-1)
        
        try:
            loss = model.criterion(reshaped_outputs, reshaped_targets)
        except RuntimeError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in loss calculation: {e}. Output shape: {outputs.shape}, Target shape for loss: {tgt_tokens_target.shape}")
            min_len = min(reshaped_outputs.shape[0], reshaped_targets.shape[0])
            if min_len > 0:
                loss = model.criterion(reshaped_outputs[:min_len], reshaped_targets[:min_len])
            else:
                loss = torch.tensor(float('nan'), device=model.device)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        device = model.device
        current_logger = logging.getLogger(__name__)

        required_input_keys = ['src_tokens', 'tgt_tokens_input', 'tgt_tokens_target']
        if inputs is None or not all(k in inputs for k in required_input_keys):
            current_logger.warning(f"Inputs missing or malformed during prediction step. Expected: {required_input_keys}. Got: {list(inputs.keys() if inputs else [])}")
            dummy_loss = torch.tensor(float('nan'), device=device)
            return (dummy_loss, None, None)
        
        src_tokens = inputs['src_tokens']
        tgt_tokens_for_model = inputs.get('tgt_tokens', inputs['tgt_tokens_input'])
        tgt_tokens_target = inputs['tgt_tokens_target']
        
        current_logger.debug(f"prediction_step shapes: src_tokens={src_tokens.shape}, tgt_tokens_for_model={tgt_tokens_for_model.shape}, tgt_tokens_target={tgt_tokens_target.shape}")
        
        try:
            with torch.no_grad():
                outputs = model(src_tokens, tgt_tokens_for_model)
                
                # Align sequence lengths before batch dimension check or loss calculation
                # outputs shape: [B, S_out, V], tgt_tokens_target shape: [B, S_tgt]
                # S_out is max_label_len, S_tgt is max_label_len - 1
                # Slice outputs to match the target length for loss calculation.
                if outputs.shape[1] > tgt_tokens_target.shape[1]: # If model output seq_len > target seq_len
                    sliced_model_outputs = outputs[:, :tgt_tokens_target.shape[1], :].contiguous()
                elif outputs.shape[1] < tgt_tokens_target.shape[1]: # Should not happen with current collator
                    current_logger.warning(f"prediction_step: model output seq_len ({outputs.shape[1]}) < target seq_len ({tgt_tokens_target.shape[1]}). Truncating target.")
                    sliced_model_outputs = outputs.contiguous()
                    tgt_tokens_target = tgt_tokens_target[:, :outputs.shape[1]].contiguous() 
                else: # Sequence lengths match
                    sliced_model_outputs = outputs.contiguous()

                tgt_tokens_target_for_loss = tgt_tokens_target # Start with the full target

                # First, ensure batch dimensions match between sliced_model_outputs and tgt_tokens_target_for_loss
                if sliced_model_outputs.shape[0] != tgt_tokens_target_for_loss.shape[0]:
                    current_logger.warning(
                        f"Batch size mismatch in prediction_step after seq slicing: sliced_model_outputs.shape[0]={sliced_model_outputs.shape[0]}, "
                        f"tgt_tokens_target_for_loss.shape[0]={tgt_tokens_target_for_loss.shape[0]}. Adjusting batch size."
                    )
                    min_batch_size = min(sliced_model_outputs.shape[0], tgt_tokens_target_for_loss.shape[0])
                    if min_batch_size == 0:
                        current_logger.error("Cannot compute loss with zero batch size after batch adjustment in prediction_step.")
                        raise ValueError("Cannot compute loss with zero batch size after batch adjustment.")
                    sliced_model_outputs = sliced_model_outputs[:min_batch_size]
                    tgt_tokens_target_for_loss = tgt_tokens_target_for_loss[:min_batch_size]
                
                # Now, sequence lengths of sliced_model_outputs and tgt_tokens_target_for_loss should match
                # from the earlier slicing logic. Batch sizes also match now.
                loss = model.criterion(sliced_model_outputs.reshape(-1, sliced_model_outputs.shape[-1]), tgt_tokens_target_for_loss.reshape(-1))
                
                if prediction_loss_only:
                    return (loss, None, None)
                
                return (loss, outputs, tgt_tokens_target)
        except Exception as e:
            current_logger.error(f"Error during prediction step: {e}", exc_info=True)
            dummy_loss = torch.tensor(float('nan'), device=device)
            return (dummy_loss, None, None)


def main():
    """
    Main function to run the training pipeline using Hugging Face's Trainer API with custom Transformer.
    Optimized for M3 Mac Pro with MPS acceleration.
    """
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Ensure output_dir is set for checkpointing and logging
    if training_args.output_dir is None:
        training_args.output_dir = "./outputs" # Default output directory
        logger.info(f"output_dir not set, defaulting to {training_args.output_dir}")
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Set up MPS acceleration for M3 Mac Pro
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable mixed precision training for better performance on MPS
        training_args.fp16 = True
        logger.info(f"Using MPS acceleration on M3 Mac Pro with mixed precision training")
        
        # Enable gradient checkpointing for better memory efficiency23
        if not hasattr(training_args, 'gradient_checkpointing') or not training_args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for better memory efficiency on M3 Mac")
            # Gradient checkpointing saves memory at the expense of some speed
            training_args.gradient_checkpointing = True
        
        # Set optimal generation parameters for MPS
        if getattr(training_args, 'generation_num_beams', None) is None or training_args.generation_num_beams < 4:
            training_args.generation_num_beams = 4
        
        # Optimize training batch size if not specifically set
        if training_args.per_device_train_batch_size > 16 and not hasattr(training_args, '_batch_size_was_set'):
            old_batch_size = training_args.per_device_train_batch_size
            training_args.per_device_train_batch_size = 16
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu")
        logger.info(f"Using device: {device}")

    # ===== IMPORTANT: FIRST LOAD TOKENIZER BEFORE ANY PROCESSING =====
    # Load and configure MarianTokenizer specifically for French-English translation
    logger.info("Loading MarianTokenizer: Helsinki-NLP/opus-mt-fr-en")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    
    # MarianTokenizer already has appropriate special tokens for translation
    vocab_size = len(tokenizer)
    logger.info(f"MarianTokenizer loaded with {vocab_size} tokens")
    logger.info(f"Special tokens: PAD={tokenizer.pad_token}, BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}")

    # ===== DEFINE PREPROCESSING FUNCTION IMMEDIATELY AFTER TOKENIZER =====
    # This ensures it is available for all downstream code
    def preprocess_function(examples):
        translations = examples["translation"]
        source_texts = [t[data_args.source_lang] for t in translations]
        target_texts = [t[data_args.target_lang] for t in translations]
        
        # Tokenize source texts with MarianTokenizer
        model_inputs = tokenizer(
            source_texts, 
            max_length=data_args.max_source_length, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target texts for MarianTokenizer
        labels = tokenizer(
            text_target=target_texts,
            max_length=data_args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Set up the labels
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Helper function to get a subset of the dataset
    def get_subset(dataset, fraction, seed=42):
        original_len = len(dataset)
        logger.info(f"get_subset called: original_len={original_len}, fraction={fraction}")
        if original_len == 0:
            logger.warning("get_subset: Input dataset is empty.")
            return dataset
        if fraction <= 0:
            logger.warning(f"get_subset: Fraction is {fraction}, returning empty dataset.")
            return dataset.select(range(0)) # Return an empty dataset of the same type
        
        n = int(original_len * fraction)
        if n == 0 and original_len > 0 and fraction > 0:
            logger.warning(f"get_subset: Calculated n=0 for non-empty dataset (len={original_len}, frac={fraction}). Selecting at least 1 sample.")
            n = 1 # Ensure at least one sample if original dataset is not empty and fraction > 0
        
        logger.info(f"get_subset: Selecting {n} samples.")
        if n > original_len:
            logger.warning(f"get_subset: Calculated n={n} is greater than original_len={original_len}. Selecting all samples.")
            n = original_len
            
        # Use a deterministic shuffle for reproducibility
        return dataset.shuffle(seed=seed).select(range(n))
    
    # Efficient data loading: use cached processed datasets if available
    processed_dir = f"data/processed/{data_args.dataset_name}_{data_args.dataset_config_name}_tok_{model_args.tokenizer_name.replace('/', '_')}"
    train_path = os.path.join(processed_dir, "train")
    eval_path = os.path.join(processed_dir, "val")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        logger.info(f"Loading tokenized datasets from disk: {processed_dir}")
        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)
        # Subsample loaded datasets to specified fraction
        train_dataset = get_subset(train_dataset, data_args.dataset_fraction)
        eval_dataset = get_subset(eval_dataset, data_args.dataset_fraction)
    else:
        logger.info(f"Loading {data_args.dataset_name} dataset with {data_args.source_lang}-{data_args.target_lang} language pair")
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        logger.info(f"Dataset loaded with {len(raw_datasets['train'])} training examples")

        # Preprocess the datasets
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation" if "validation" in raw_datasets else "test"]
        # Subsample to specified fraction before tokenization to save space
        train_dataset = get_subset(train_dataset, data_args.dataset_fraction)
        eval_dataset = get_subset(eval_dataset, data_args.dataset_fraction)
        
        # Optimize preprocessing for M3 Mac
        import os as _os
        optimal_workers = data_args.preprocessing_num_workers or max(1, _os.cpu_count() - 1)
        logger.info(f"Using {optimal_workers} workers for dataset preprocessing")
        
        logger.info("Preprocessing training dataset...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1024,
            num_proc=optimal_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        
        logger.info("Preprocessing validation dataset...")
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1024,
            num_proc=optimal_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        
        logger.info(f"Processed dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        # Save processed datasets to disk for future fast loading
        logger.info(f"Saving processed datasets to {processed_dir}")
        os.makedirs(processed_dir, exist_ok=True)
        train_dataset.save_to_disk(train_path)
        eval_dataset.save_to_disk(eval_path)

    # Initialize the custom Transformer model with MarianTokenizer's vocabulary
    logger.info("Initializing custom Transformer model...")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,  # Same vocabulary for source and target
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        dim_embedding=model_args.dim_embedding,
        num_heads=model_args.num_heads,
        dim_feedfordward=model_args.dim_feedforward,
        dropout=model_args.dropout,
        pad_idx=tokenizer.pad_token_id  # Use MarianTokenizer's pad token ID
    )
    logger.info(f"Model vocabulary size: {vocab_size} (using MarianTokenizer's vocabulary)")
    
    # Move model to appropriate device (MPS for M3 Mac)
    model.to(device)
    logger.info(f"Model initialized and moved to {device}")
        
    # Set up M3 Mac-specific training optimizations
    if device.type == 'mps':
        # Enable these settings for better performance on M3 Mac
        if not hasattr(training_args, 'gradient_checkpointing') or not training_args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for better memory efficiency on M3 Mac")
            # Gradient checkpointing saves memory at the expense of some speed
            training_args.gradient_checkpointing = True
            
        # Set optimal generation parameters for MPS
        if getattr(training_args, 'generation_num_beams', None) is None or training_args.generation_num_beams < 4:
            training_args.generation_num_beams = 4
            logger.info(f"Setting beam search to {training_args.generation_num_beams} beams for better translation quality")
            
        # Optimize training batch size if not specifically set
        if training_args.per_device_train_batch_size > 16 and not hasattr(training_args, '_batch_size_was_set'):
            old_batch_size = training_args.per_device_train_batch_size
            training_args.per_device_train_batch_size = 16
            logger.info(f"Adjusted batch size from {old_batch_size} to {training_args.per_device_train_batch_size} for M3 Mac")
    
    # Create data collator for batching with device information
    data_collator = CustomDataCollator(tokenizer, tokenizer.pad_token_id, device)
        
    # Update training arguments for better learning only if not explicitly set via command line
    # Check if num_train_epochs was provided in command line arguments
    import sys
    epochs_in_args = any('--num_train_epochs' in arg for arg in sys.argv)
    
    if not epochs_in_args and not hasattr(training_args, '_epochs_were_set'):
        # Only set default epochs if not specified by user
        training_args.num_train_epochs = 15  # Increase from default to 15
        logger.info(f"Setting training epochs to {training_args.num_train_epochs} for better translation quality")
    else:
        logger.info(f"Using user-specified epochs: {training_args.num_train_epochs}")
        
    # Set up early stopping and learning rate scheduler if not explicitly configured
    # Check if these were provided in command line arguments
    best_model_in_args = any('--load_best_model_at_end' in arg for arg in sys.argv)
    scheduler_in_args = any('--lr_scheduler_type' in arg for arg in sys.argv)
    warmup_in_args = any('--warmup_steps' in arg for arg in sys.argv)
    
    # Only enable early stopping if not specified by user
    if not best_model_in_args and not training_args.load_best_model_at_end:
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "loss"
        training_args.greater_is_better = False
        logger.info("Enabled loading best model at end of training based on loss")
    
    # Only change scheduler if not specified by user
    if not scheduler_in_args and training_args.lr_scheduler_type == "linear":
        training_args.lr_scheduler_type = "cosine"
        logger.info(f"Set learning rate scheduler to cosine")
        
    # Only set warmup steps if not specified by user
    if not warmup_in_args and not training_args.warmup_steps:
        training_args.warmup_steps = 500
        logger.info(f"Set warmup steps to {training_args.warmup_steps}")
        
    logger.info("Loading BLEU metric...")
    try:
        bleu_metric_for_eval = evaluate.load("bleu")
    except Exception as e:
        logger.error(f"Failed to load BLEU metric: {e}. Evaluation will not have BLEU scores.")
        bleu_metric_for_eval = None

    def compute_metrics(eval_preds):
        # tokenizer should be in the scope of main() or passed appropriately
        # bleu_metric_for_eval is from the scope of main()
        
        preds_logits, label_ids = eval_preds
        
        if preds_logits is None:
            logger.error("Predictions (logits) are None in compute_metrics.")
            return {"eval_bleu": 0.0, "gen_len": 0.0, "eval_bp": 0.0}

        if np.isnan(preds_logits).any():
            logger.warning("NaN found in predictions (logits) in compute_metrics. Skipping.")
            return {"eval_bleu": 0.0, "gen_len": 0.0, "eval_bp": 0.0}

        try:
            # Get actual token ID predictions from logits
            predictions_token_ids = np.argmax(preds_logits, axis=-1)
        except Exception as e:
            logger.error(f"Error in np.argmax (preds_logits shape: {preds_logits.shape if hasattr(preds_logits, 'shape') else 'N/A'}): {e}")
            return {"eval_bleu": 0.0, "gen_len": 0.0, "eval_bp": 0.0}

        # Replace -100 in the label_ids as we can't decode them.
        processed_label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        try:
            decoded_preds_raw = tokenizer.batch_decode(predictions_token_ids, skip_special_tokens=True)
            decoded_labels_raw = tokenizer.batch_decode(processed_label_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error decoding predictions/labels in compute_metrics: {e}")
            return {"eval_bleu": 0.0, "gen_len": 0.0, "eval_bp": 0.0}

        # Prepare for BLEU: list of lists for references, list of strings for predictions
        # Strip whitespace as it can affect BLEU and is often not desired.
        decoded_labels_for_bleu = [[label.strip()] for label in decoded_labels_raw]
        decoded_preds_for_bleu = [pred.strip() for pred in decoded_preds_raw]
        
        bleu_score_val = 0.0
        brevity_penalty_val = 0.0

        if bleu_metric_for_eval is not None:
            # Filter out pairs where either prediction or reference is empty after stripping,
            # as this can cause issues or lead to misleading BLEU scores (e.g., 0/0 division).
            valid_pred_strings = []
            valid_ref_lists = []
            for pred_str, ref_list_str in zip(decoded_preds_for_bleu, decoded_labels_for_bleu):
                if pred_str and ref_list_str[0]: # Ensure both pred and its single ref are non-empty
                    valid_pred_strings.append(pred_str)
                    valid_ref_lists.append(ref_list_str)
            
            if not valid_pred_strings: # Or check valid_ref_lists, they should have same length
                logger.warning("No valid (non-empty) prediction/reference pairs for BLEU calculation after stripping.")
            else:
                try:
                    # logger.debug(f"Calculating BLEU with {len(valid_pred_strings)} pairs.")
                    # logger.debug(f"Sample pred for BLEU: '{valid_pred_strings[0]}'")
                    # logger.debug(f"Sample ref for BLEU: {valid_ref_lists[0]}")
                    bleu_result = bleu_metric_for_eval.compute(predictions=valid_pred_strings, references=valid_ref_lists)
                    if bleu_result:
                        bleu_score_val = bleu_result.get('bleu', 0.0)
                        brevity_penalty_val = bleu_result.get('brevity_penalty', 0.0)
                        # logger.debug(f"Full BLEU result: {bleu_result}")
                except Exception as e:
                    logger.error(f"Error computing BLEU score with bleu_metric_for_eval: {e}")
                    # logger.error(f"Preds for BLEU: {valid_pred_strings[:2]}")
                    # logger.error(f"Refs for BLEU: {valid_ref_lists[:2]}")

        else:
            logger.warning("BLEU metric loader not available. BLEU score will be 0.")

        # Calculate mean generated length from original (pre-strip) decoded_preds
        # Using decoded_preds_raw as stripping might change length if only spaces were predicted.
        gen_len = np.mean([len(p.split()) for p in decoded_preds_raw]) if decoded_preds_raw else 0.0

        final_metrics = {
            'bleu': round(float(bleu_score_val), 4),
            'gen_len': round(float(gen_len), 4) if not np.isnan(gen_len) else 0.0,
            'bp': round(float(brevity_penalty_val), 4)
        }
        
        # logger.info(f"Computed metrics: {final_metrics}") # For debugging
        return final_metrics
    
    # Initialize Trainer with detailed logging
    logger.info("Initializing Trainer with the following settings:")
    logger.info(f"  Model: Custom Transformer with {model_args.num_encoder_layers} encoder and {model_args.num_decoder_layers} decoder layers")
    logger.info(f"  Device: {device} ({'Mixed precision' if training_args.fp16 else 'Full precision'})")
    logger.info(f"  Training batch size: {training_args.per_device_train_batch_size} per device")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # Add custom callbacks for checkpoint handling
    callbacks = [SaveLastCheckpointCallback()]
    
    # CRITICAL: Set remove_unused_columns=False to prevent the dataset column filtering
    # This allows our model to receive all columns even if they don't match the forward method signature
    training_args.remove_unused_columns = False
    
    # Disable gradient checkpointing as our custom model doesn't implement the gradient_checkpointing_enable method
    training_args.gradient_checkpointing = False
    
    # Disable pin_memory when using MPS (not supported on Apple Silicon)
    if device.type == "mps":
        training_args.dataloader_pin_memory = False
    
    # Initialize the trainer with all necessary components
    trainer = TransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks  # Add custom callbacks for checkpoint management
    )
    
    # Training with detailed progress reporting
    if training_args.do_train:
        logger.info("Starting training...")
        # Record start time for performance measurement
        import time
        start_time = time.time()
        
        # Check for existing checkpoints to resume training
        logger.info(f"Checking for checkpoint. training_args.output_dir='{training_args.output_dir}', training_args.overwrite_output_dir is {training_args.overwrite_output_dir}")
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-last")
        if os.path.exists(checkpoint_dir) and not training_args.overwrite_output_dir:
            logger.info(f"Resuming training from checkpoint in {checkpoint_dir}")
            train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            logger.info(f"Starting training from scratch. Reason: checkpoint_dir '{checkpoint_dir}' exists? {os.path.exists(checkpoint_dir)}. overwrite_output_dir? {training_args.overwrite_output_dir}.")
            train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Save the model and training metrics
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model()  # Saves the model and tokenizer to training_args.output_dir (e.g. ./outputs/)

        # Explicitly save to 'checkpoint-last' for reliable resumption
        last_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-last")
        logger.info(f"Explicitly saving final model and tokenizer to {last_checkpoint_dir} for resumption.")
        try:
            # Save model and tokenizer
            trainer.save_model(last_checkpoint_dir)
            # Save the Trainer state (for resumption)
            trainer.state.save_to_json(os.path.join(last_checkpoint_dir, "trainer_state.json"))
            # Save training arguments (for completeness)
            torch.save(training_args, os.path.join(last_checkpoint_dir, "training_args.bin"))
            logger.info(f"Successfully saved checkpoint to {last_checkpoint_dir}.")
        except Exception as e:
            logger.error(f"Failed to save to {last_checkpoint_dir}: {e}", exc_info=True)
        
        # Save model in a format optimized for Mac M3 inference
        try:
            logger.info("Saving model optimized for Mac M3 inference...")
            optimized_model_path = os.path.join(training_args.output_dir, "optimized_model")
            os.makedirs(optimized_model_path, exist_ok=True)
            
            # Save quantized model for faster inference on Mac M3
            with torch.no_grad():
                # Convert to fp16 for better performance on MPS
                model_fp16 = model.half()
                # Save optimized model
                torch.save({
                    'model_state_dict': model_fp16.state_dict(),
                    'vocab_size': vocab_size,
                    'model_config': {
                        'dim_embedding': model_args.dim_embedding,
                        'num_heads': model_args.num_heads,
                        'num_encoder_layers': model_args.num_encoder_layers,
                        'num_decoder_layers': model_args.num_decoder_layers,
                        'dim_feedforward': model_args.dim_feedforward,
                        'dropout': model_args.dropout
                    }
                }, os.path.join(optimized_model_path, "model_optimized.pt"))
                
                # Save tokenizer for easy loading
                tokenizer.save_pretrained(optimized_model_path)
                
            logger.info(f"Optimized model saved to {optimized_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save optimized model: {e}")
        
        metrics = train_result.metrics
        metrics["training_time"] = training_time
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation with detailed reporting and robust error handling
    if training_args.do_eval:
        logger.info("Starting evaluation...")
        try:
            # Verify evaluation dataset is properly prepared
            if eval_dataset is None or len(eval_dataset) == 0:
                logger.warning("Evaluation dataset is empty or None. Skipping evaluation.")
            else:
                # Check a sample from the evaluation dataset to ensure it's properly formatted
                sample = eval_dataset[0]
                logger.info(f"Evaluation dataset sample keys: {list(sample.keys())}")
                
                # Ensure required keys are present
                required_keys = ['input_ids', 'attention_mask', 'labels']
                if not all(k in sample for k in required_keys):
                    logger.warning(f"Evaluation dataset missing required keys. Found: {list(sample.keys())}")
                    logger.warning("Will attempt evaluation but it may fail.")
                
                # Run evaluation with proper error handling
                metrics = trainer.evaluate(
                    max_length=data_args.max_target_length, 
                    num_beams=training_args.generation_num_beams,
                    metric_key_prefix="eval"
                )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            logger.error("Continuing without evaluation. The model was still trained and saved.")
            # Create empty metrics to avoid further errors
            metrics = {"eval_loss": float('nan')}
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        
        # Show sample translations with configurable beam search
        logger.info("\nSample translations:")
        samples = eval_dataset.select(range(min(5, len(eval_dataset))))
        
        # Use configured beam size or default to 4
        num_beams = getattr(training_args, 'generation_num_beams', 4)
        logger.info(f"Generating translations with beam size: {num_beams}")
        
        for i, sample in enumerate(samples):
            # Move tensors to the device
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
            
            # Generate translation with beam search - optimized for MarianTokenizer
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    max_length=data_args.max_target_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    decoder_start_token_id=tokenizer.pad_token_id, # Marian models often use pad_token_id as the decoder_start_token_id
                    length_penalty=0.6  # Favor slightly longer translations (better for BLEU)
                )
            
            # Decode the translation
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            source = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            reference = tokenizer.decode(sample["labels"], skip_special_tokens=True)
            
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Source (French): {source}")
            logger.info(f"Reference (English): {reference}")
            logger.info(f"Model Translation: {translation}")
    
    return trainer


def load_optimized_model(model_path, device=None):
    """
    Helper function to load a model optimized for inference.
    
    Args:
        model_path: Path to the optimized model directory
        device: Device to load the model on (default: auto-detect)
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                           "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load optimized model file
    checkpoint = torch.load(os.path.join(model_path, "model_optimized.pt"), map_location=device)
    
    # Extract model configuration
    config = checkpoint['model_config']
    vocab_size = checkpoint['vocab_size']
    
    # Initialize model with the same configuration
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_embedding=config['dim_embedding'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedfordward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


if __name__ == "__main__":
    main()
