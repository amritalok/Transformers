#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model analysis script for the trained Transformer model.

This script analyzes the trained model by examining:
1. Encoder representations for French inputs
2. Vocabulary embeddings and clustering
3. Attention patterns

This helps verify if the model learned meaningful patterns even if
full translation generation is challenging.
"""

import os
import torch
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import MarianTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_optimized_model(model_path, device=None):
    """
    Helper function to load a model optimized for inference.
    """
    # Import here to avoid circular imports
    from model.Transformer import Transformer
    
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                           "cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading model on device: {device}")
    
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
    
    return model, tokenizer, config


def analyze_encoder_representations(model, tokenizer, sentences):
    """
    Analyze the encoder representations for a set of input sentences.
    """
    device = next(model.parameters()).device
    
    logger.info("\nAnalyzing encoder representations...")
    
    # Store all encodings for visualization
    all_encodings = []
    all_tokens = []
    
    for sentence in sentences:
        # Tokenize input
        tokenized = tokenizer(sentence, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)
        
        # Decode tokens for visualization
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        # Get encoder representations
        with torch.no_grad():
            # Process through embedding and encoder directly
            src_embeddings = model.src_embedding(input_ids) * model.embedding_scale
            src_seq_len = input_ids.size(1)
            src_positions = model.positional_encoding[:, :src_seq_len, :]
            src_embeddings = model.dropout(src_embeddings + src_positions)
            
            # Get encoder output
            encoder_output = model.encoder(src_embeddings)
            
            # Move to CPU for analysis
            encoder_output = encoder_output.cpu().numpy()
        
        # Log info
        logger.info(f"Input: {sentence}")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Encoder output shape: {encoder_output.shape}")
        
        # Store for visualization
        all_encodings.append(encoder_output[0])
        all_tokens.extend([(sentence, token) for token in tokens])
    
    # Visualize encoder representations using PCA
    visualize_encodings(all_encodings, all_tokens)


def visualize_encodings(encodings, tokens, n_components=2):
    """
    Visualize encodings using PCA projection.
    """
    # Flatten all encodings
    flattened = np.vstack([enc for enc in encodings])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flattened)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot points
    for i, ((sentence, token), point) in enumerate(zip(tokens, reduced)):
        plt.scatter(point[0], point[1], alpha=0.7)
        plt.annotate(token, (point[0], point[1]), fontsize=8)
    
    plt.title(f"PCA Projection of Encoder Representations")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.tight_layout()
    
    # Save figure
    output_dir = "./outputs/analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "encoder_representations.png"), dpi=300)
    logger.info(f"Saved visualization to {os.path.join(output_dir, 'encoder_representations.png')}")


def analyze_embedding_similarities(model, tokenizer):
    """
    Analyze embedding similarities between source and target languages.
    """
    logger.info("\nAnalyzing embedding similarities...")
    
    # Get embeddings
    src_embeddings = model.src_embedding.weight.detach().cpu().numpy()
    tgt_embeddings = model.tgt_embedding.weight.detach().cpu().numpy()
    
    # Check if they're the same
    is_shared = np.allclose(src_embeddings, tgt_embeddings)
    logger.info(f"Source and target embeddings are shared: {is_shared}")
    
    # Get some common words to examine
    common_words = [
        "hello", "bonjour", "world", "monde", "thank", "merci",
        "yes", "oui", "no", "non", "good", "bon"
    ]
    
    # Try to find token IDs for these words
    found_tokens = []
    for word in common_words:
        try:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if token_ids:
                found_tokens.append((word, token_ids[0]))
        except:
            pass
    
    # Compare embeddings for these tokens
    logger.info("\nCommon word embedding analysis:")
    for word, token_id in found_tokens:
        if token_id < len(src_embeddings):
            src_emb = src_embeddings[token_id]
            tgt_emb = tgt_embeddings[token_id]
            similarity = np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb))
            logger.info(f"Word: {word}, Token ID: {token_id}, Embedding similarity: {similarity:.4f}")


def analyze_model_params(model, config):
    """
    Analyze model parameters and architecture.
    """
    logger.info("\nModel Architecture Analysis:")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Architecture summary
    logger.info(f"\nArchitecture Configuration:")
    logger.info(f"Embedding dimension: {config['dim_embedding']}")
    logger.info(f"Number of attention heads: {config['num_heads']}")
    logger.info(f"Number of encoder layers: {config['num_encoder_layers']}")
    logger.info(f"Number of decoder layers: {config['num_decoder_layers']}")
    logger.info(f"Feed-forward dimension: {config['dim_feedforward']}")
    logger.info(f"Dropout rate: {config['dropout']}")
    
    # Parameter magnitudes (to check for potential vanishing/exploding gradients)
    logger.info("\nParameter magnitude analysis:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: mean={param.data.abs().mean().item():.6f}, max={param.data.abs().max().item():.6f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze a trained Transformer model")
    parser.add_argument("--model_path", type=str, default="./outputs/optimized_model", 
                      help="Path to the optimized model")
    args = parser.parse_args()
    
    # Load the model and tokenizer
    model, tokenizer, config = load_optimized_model(args.model_path)
    
    # Log model information
    logger.info(f"Loaded model from {args.model_path}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Define sample sentences for testing
    sample_sentences = [
        "Bonjour, comment ça va?",
        "Je suis très heureux de vous rencontrer.",
        "Quelle est la capitale de la France?",
        "L'intelligence artificielle transforme notre monde.",
        "J'aimerais réserver une table pour deux personnes ce soir."
    ]
    
    # Analyze model parameters
    analyze_model_params(model, config)
    
    # Analyze embedding similarities
    analyze_embedding_similarities(model, tokenizer)
    
    # Analyze encoder representations
    analyze_encoder_representations(model, tokenizer, sample_sentences)
    
    logger.info("\nAnalysis completed!")


if __name__ == "__main__":
    main()
