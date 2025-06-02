#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer training script using Hugging Face's Trainer API.

This script uses the custom Transformer implementation from the 'model' directory
while leveraging Hugging Face's datasets and Trainer API for efficient training.
Optimized for M3 Mac Pro with Metal Performance Shaders (MPS) acceleration.
"""

import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

# Hugging Face imports
import evaluate
from datasets import load_dataset
from transformers import (
    MarianTokenizer,  # Specialized tokenizer for translation tasks
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser
)

# Import custom Transformer model
from model.Transformer import Transformer


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
        # Collect input_ids, attention_masks, and labels from features
        input_ids = torch.stack([feature['input_ids'] for feature in features])
        attention_mask = torch.stack([feature['attention_mask'] for feature in features])
        labels = torch.stack([feature['labels'] for feature in features])
        
        # For teacher forcing, we need target_input (shifted right) and target_output
        # Target input is the labels but with the last token removed (we'll add BOS at the beginning)
        target_input = labels[:, :-1].clone()
        
        # Add BOS token at the beginning
        bos_tensor = torch.full(
            (target_input.size(0), 1), 
            self.tokenizer.bos_token_id, 
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
    def compute_loss(self, model, inputs, return_outputs=False):
        # Move inputs to the model's device
        device = model.device
        src_tokens = inputs['src_tokens'].to(device)
        tgt_tokens_input = inputs['tgt_tokens_input'].to(device)
        tgt_tokens_target = inputs['tgt_tokens_target'].to(device)
        
        # Forward pass through the model
        outputs = model(src_tokens, tgt_tokens_input)
        
        # Reshape predictions and targets for loss calculation
        loss = model.criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_tokens_target.reshape(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Move inputs to the model's device
        device = model.device
        src_tokens = inputs['src_tokens'].to(device)
        tgt_tokens_input = inputs['tgt_tokens_input'].to(device)
        tgt_tokens_target = inputs['tgt_tokens_target'].to(device)
        
        with torch.no_grad():
            # Forward pass through the model
            outputs = model(src_tokens, tgt_tokens_input)
            loss = model.criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_tokens_target.reshape(-1))
            
            if prediction_loss_only:
                return (loss, None, None)
            
            # Return loss, logits, and labels
            return (loss, outputs, tgt_tokens_target)


def main():
    """
    Main function to run the training pipeline using Hugging Face's Trainer API with custom Transformer.
    Optimized for M3 Mac Pro with MPS acceleration.
    """
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
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
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu")
        logger.info(f"Using device: {device}")
    
    # Efficient data loading: use cached processed datasets if available
    import os
    from datasets import load_from_disk
    processed_dir = f"data/processed/{data_args.dataset_name}_{data_args.dataset_config_name}_tok_{model_args.tokenizer_name.replace('/', '_')}"
    train_path = os.path.join(processed_dir, "train")
    eval_path = os.path.join(processed_dir, "val")

    def get_subset(dataset, fraction, seed=42):
        n = int(len(dataset) * fraction)
        # Use a deterministic shuffle for reproducibility
        return dataset.shuffle(seed=seed).select(range(n))

    # Load and configure MarianTokenizer specifically for French-English translation
    logger.info("Loading MarianTokenizer: Helsinki-NLP/opus-mt-fr-en")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    
    # MarianTokenizer already has appropriate special tokens for translation
    vocab_size = len(tokenizer)
    logger.info(f"MarianTokenizer loaded with {vocab_size} tokens")
    logger.info(f"Special tokens: PAD={tokenizer.pad_token}, BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}")

    # Define the preprocessing function before it's used
    def preprocess_function(examples):
        translations = examples["translation"]
        source_texts = [t[data_args.source_lang] for t in translations]
        target_texts = [t[data_args.target_lang] for t in translations]
        model_inputs = tokenizer(
            source_texts, 
            max_length=data_args.max_source_length, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenizer(
            text_target=target_texts,
            max_length=data_args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if os.path.exists(train_path) and os.path.exists(eval_path):
        logger.info(f"Loading tokenized datasets from disk: {processed_dir}")
        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)
        # Subsample loaded datasets to 25%
        train_dataset = get_subset(train_dataset, data_args.dataset_fraction)
        eval_dataset = get_subset(eval_dataset, data_args.dataset_fraction)
    else:
        logger.info(f"Loading {data_args.dataset_name} dataset with {data_args.source_lang}-{data_args.target_lang} language pair")
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        logger.info(f"Dataset loaded with {len(raw_datasets['train'])} training examples")

        # Preprocess the datasets
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation" if "validation" in raw_datasets else "test"]
        # Subsample to 25% before tokenization to save space
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

    
    # Note: tokenizer and preprocess_function are already defined above

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
    
    # The preprocess_function is already defined above at line 307
    
    # Preprocess the datasets
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation" if "validation" in raw_datasets else "test"]
    
    # Limit the number of samples if specified
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    
    # Optimize preprocessing for M3 Mac
    import os
    # Determine optimal number of workers for M3 Mac - use one fewer than available cores
    optimal_workers = data_args.preprocessing_num_workers or max(1, os.cpu_count() - 1)
    logger.info(f"Using {optimal_workers} workers for dataset preprocessing")
    
    # Apply preprocessing with larger batch sizes for efficiency
    logger.info("Preprocessing training dataset...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1024,  # Larger batch size for faster preprocessing
        num_proc=optimal_workers,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    
    logger.info("Preprocessing validation dataset...")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1024,  # Larger batch size for faster preprocessing
        num_proc=optimal_workers,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
    
    logger.info(f"Processed dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Create data collator for batching with device information
    data_collator = CustomDataCollator(tokenizer, tokenizer.pad_token_id, device)
    
    # Metric for evaluation
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Ignore padding in labels (-100)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Get predictions (argmax)
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions = np.argmax(preds, axis=-1)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU score
        result = metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        # Add mean generated length
        result["gen_len"] = np.mean([len(pred.split()) for pred in decoded_preds])
        
        return {k: round(v, 4) for k, v in result.items()}
    
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
    
    # Initialize Trainer with detailed logging
    logger.info("Initializing Trainer with the following settings:")
    logger.info(f"  Model: Custom Transformer with {model_args.num_encoder_layers} encoder and {model_args.num_decoder_layers} decoder layers")
    logger.info(f"  Device: {device} ({'Mixed precision' if training_args.fp16 else 'Full precision'})")
    logger.info(f"  Training batch size: {training_args.per_device_train_batch_size} per device")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    trainer = TransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Changed from 'tokenizer' to 'processing_class' to fix deprecation warning
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training with detailed progress reporting
    if training_args.do_train:
        logger.info("Starting training...")
        # Record start time for performance measurement
        import time
        start_time = time.time()
        
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Save the model and training metrics
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model()  # Saves the tokenizer too
        
        metrics = train_result.metrics
        metrics["training_time"] = training_time
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation with detailed reporting
    if training_args.do_eval:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate(
            max_length=data_args.max_target_length, 
            num_beams=training_args.generation_num_beams,
            metric_key_prefix="eval"
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        # Show sample translations
        logger.info("\nSample translations:")
        samples = eval_dataset.select(range(min(5, len(eval_dataset))))
        for i, sample in enumerate(samples):
            # Move tensors to the device
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
            
            # Generate translation with beam search - optimized for MarianTokenizer
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    max_length=data_args.max_target_length,
                    num_beams=4,
                    early_stopping=True,
                    decoder_start_token_id=tokenizer.bos_token_id  # Ensure proper start token for MarianTokenizer
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


if __name__ == "__main__":
    main()
