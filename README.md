# Transformers

A PyTorch implementation of the "Attention is All You Need" Transformer model for neural machine translation, optimized for M3 Mac hardware.

## Overview

This project implements the Transformer architecture as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The implementation uses PyTorch and HuggingFace's Transformers and Datasets libraries for efficient training and evaluation.

### Features

- Complete custom implementation of the Transformer architecture
- Multi-head attention mechanism
- Positional encoding
- Optimized for M3 Mac with MPS acceleration
- Uses HuggingFace's Seq2SeqTrainer for training and evaluation
- Memory-efficient with gradient checkpointing
- Mixed precision (FP16) training for faster computation
- Automatic dataset caching and checkpoint handling
- Model quantization for efficient inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Transformers.git
cd Transformers

# Install dependencies using Poetry
poetry install
```

## Requirements

This project requires:

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers 4.28+
- HuggingFace Datasets 2.12+
- HuggingFace Evaluate 0.4+

The full list of dependencies can be found in `pyproject.toml`.

## Project Structure

```
Transformers/
├── data/                # Dataset storage
├── src/
│   ├── model/           # Custom Transformer implementation
│   │   ├── Decoder.py
│   │   ├── DecoderLayer.py
│   │   ├── Encoder.py
│   │   ├── EncoderLayer.py
│   │   ├── MultiHeadAttention.py
│   │   ├── Residual.py
│   │   ├── SelfAttention.py
│   │   └── Transformer.py
│   ├── train_fixed.py   # Original training script
│   └── train_revised.py # Enhanced training script with optimizations
├── outputs/             # Model checkpoints and outputs
├── pyproject.toml      # Poetry dependency definitions
└── README.md
```

## Usage

### Training the Model

You can train the model using the `train_revised.py` script, which includes optimizations for M3 Mac hardware:

```bash
poetry run python src/train_revised.py \
    --output_dir ./outputs \
    --do_train \
    --do_eval \
    --dataset_name wmt14 \
    --dataset_config_name fr-en \
    --dataset_fraction 0.25 \
    --source_lang fr \
    --target_lang en \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 100 \
    --fp16
```

### Key Parameters

You can customize the model and training with these parameters:

#### Model Configuration
- `--dim_embedding`: Dimension of token embeddings (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 6)
- `--num_decoder_layers`: Number of decoder layers (default: 6)
- `--dim_feedforward`: Dimension of feed-forward network (default: 2048)
- `--dropout`: Dropout rate (default: 0.1)
- `--tokenizer_name`: HuggingFace tokenizer name (default: "Helsinki-NLP/opus-mt-fr-en")

#### Dataset Configuration
- `--dataset_name`: HuggingFace dataset name (default: "wmt14")
- `--dataset_config_name`: Dataset configuration (default: "fr-en")
- `--dataset_fraction`: Fraction of dataset to use (default: 0.25)
- `--max_source_length`: Maximum source sequence length (default: 128)
- `--max_target_length`: Maximum target sequence length (default: 128)

#### Training Configuration
- `--per_device_train_batch_size`: Batch size per device for training (default: 16)
- `--gradient_accumulation_steps`: Number of updates steps to accumulate before backward pass (default: 2)
- `--learning_rate`: Initial learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--fp16`: Enable mixed precision training (recommended for M3 Mac)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory

### Using a Small Dataset for Testing

By default, the script uses a fraction of the dataset (25%) for both training and evaluation to allow for quick testing and iteration. You can adjust this using the `--dataset_fraction` parameter.

For full training on the complete dataset, set `--dataset_fraction 1.0`.

### Resuming Training from Checkpoints

The training script automatically saves checkpoints and can resume training from the last checkpoint:

```bash
poetry run python src/train_revised.py \
    --output_dir ./outputs \
    --do_train \
    --do_eval
```

If a checkpoint exists in `./outputs/checkpoint-last`, training will automatically resume from there.

### Inference with the Trained Model

You can use the trained model for inference in your Python code:

```python
from src.train_revised import load_optimized_model

# Load the optimized model (automatically selects MPS on Mac M3)
model, tokenizer = load_optimized_model("./outputs/optimized_model")

# Prepare input text
input_text = "Bonjour, comment ça va?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# Generate translation with beam search
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=tokenizer.bos_token_id
    )

# Decode the translation
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {input_text}")
print(f"Translation: {translation}")
```

## Model Architecture

The Transformer implementation follows the architecture described in the original paper:

1. **Encoder**: Processes the input sequence
   - Multi-head self-attention
   - Position-wise feed-forward networks
   - Residual connections and layer normalization

2. **Decoder**: Generates the output sequence
   - Masked multi-head self-attention
   - Multi-head cross-attention to encoder outputs
   - Position-wise feed-forward networks
   - Residual connections and layer normalization

## M3 Mac Optimizations

- **MPS Acceleration**: Automatically uses Metal Performance Shaders for GPU acceleration on M3 Macs
- **Mixed Precision Training**: Uses FP16 for faster computation and reduced memory usage
- **Gradient Checkpointing**: Reduces memory requirements during training by recomputing activations
- **Optimized Batch Size**: Automatically selects appropriate batch size for M3 Mac
- **Efficient Workers**: Uses optimal number of workers for dataset preprocessing
- **Cached Datasets**: Preprocessed datasets are saved to disk to avoid redundant processing
- **Optimized Model Saving**: Half-precision model saved specifically for efficient inference on M3 Mac

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/index)
