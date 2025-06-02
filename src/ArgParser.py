import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Train a Transformer model using Hugging Face datasets")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", default="wmt14", help='Name of the Hugging Face dataset to use (e.g., wmt14, opus100)')
    parser.add_argument("--src_lang", default="en", help='Source language code (e.g., en)')
    parser.add_argument("--tgt_lang", default="fr", help='Target language code (e.g., fr)')
    parser.add_argument("--max_length", default=128, type=int, help='Maximum sequence length for tokenization')
    
    # Training parameters
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size for training')
    parser.add_argument("--max_epochs", default=10, type=int, help='Maximum number of training epochs')
    parser.add_argument("--steps", default=1000, type=int, help='Steps per epoch')
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU/MPS for training")
    
    # Logging parameters
    parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb", None], help='Logger to use')
    parser.add_argument("--log_every_n_steps", default=10, type=int, help='Log every N steps')
    
    return parser