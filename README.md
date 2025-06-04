# Custom Transformer Implementation for Neural Machine Translation

## 1. Overview

This project implements the Transformer model, as introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al., from scratch using PyTorch. It leverages Hugging Face's `datasets` library for data loading and preprocessing, `tokenizers` for text tokenization, and the `Trainer` API for orchestrating the training and evaluation loops. The primary goal is to build a neural machine translation (NMT) system, initially demonstrated for French-to-English translation, with optimizations for running on Apple Silicon (Mac M3 Pro with MPS).

The project emphasizes:
-   A deep understanding of the Transformer architecture by building its components.
-   Efficient data handling pipelines, including caching and subsetting for rapid development.
-   Reliable checkpointing and resumption of training.
-   Clear separation of concerns between model, data, and training logic.

## 2. Features

-   **Custom Transformer Model:**
    -   Complete from-scratch implementation of the Transformer encoder-decoder architecture using PyTorch `nn.Module`.
    -   Includes core components: Multi-Head Attention (self-attention and cross-attention), Position-wise Feed-Forward Networks, Positional Encoding, and Residual Connections with Layer Normalization.
-   **Data Pipeline & Handling:**
    -   Utilizes Hugging Face `datasets` to load standard translation datasets (e.g., WMT14 fr-en).
    -   Supports dataset subsetting via `--dataset_fraction` for quick iteration and debugging.
    -   Efficient tokenization using Hugging Face `tokenizers` (e.g., `MarianTokenizer`).
    -   Automated caching of preprocessed (tokenized) datasets to `./data/processed/` to save time on subsequent runs.
-   **Training & Evaluation:**
    -   Managed by Hugging Face `Trainer` API (`Seq2SeqTrainer` customized as `TransformerTrainer`).
    -   Supports evaluation with BLEU score and other relevant metrics.
    -   Generates sample translations for qualitative assessment of model performance.
-   **Optimizations for Mac M3 Pro (Apple Silicon):**
    -   Metal Performance Shaders (MPS) acceleration is auto-detected and utilized.
    -   Considerations for mixed precision training (noting that `fp16` is primarily for CUDA; MPS support for low-precision types is handled in the script).
    -   Gradient checkpointing (configurable via `TrainingArguments`, though requires explicit model support if not using Hugging Face's built-in models).
    -   Efficient worker utilization for dataset preprocessing.
-   **Checkpointing & Resumption:**
    -   Robust saving of model, tokenizer, trainer state, and training arguments to a `./outputs/checkpoint-last/` directory.
    -   Automatic resumption from the last checkpoint if available, enabling continuation of interrupted training runs.
-   **Custom Components for Advanced Control:**
    -   `CustomDataCollator`: Handles dynamic padding and prepares batch inputs specifically for teacher forcing in the custom Transformer.
    -   `TransformerTrainer`: A subclass of `Seq2SeqTrainer` with a customized `prediction_step` for tailored evaluation loss calculation if needed.
-   **Model Quantization & Optimized Inference (Conceptual):**
    -   Includes logic to save the model in a format (`./outputs/optimized_model/`) that includes configuration, potentially suitable for future quantization and efficient inference.

## 3. Project Structure

```
.
├── data/
│   └── processed/                 # Cached tokenized datasets
├── outputs/
│   ├── checkpoint-xxxx/           # Trainer's default epoch/step-based checkpoints
│   ├── checkpoint-last/           # Explicitly saved final checkpoint for reliable resumption
│   └── optimized_model/           # Model saved in a specific format for inference
├── src/
│   ├── model/
│   │   ├── Transformer.py         # Core Transformer model definition
│   │   ├── EncoderLayer.py
│   │   ├── DecoderLayer.py
│   │   ├── MultiHeadAttention.py
│   │   ├── PositionwiseFeedForward.py
│   │   ├── PositionalEncoding.py
│   │   └── Residual.py
│   ├── train_revised.py         # Main training and evaluation script
│   └── (other utility scripts if any)
├── poetry.lock                    # Poetry lock file
├── pyproject.toml                 # Poetry project configuration and dependencies
└── README.md                      # This file
```

## 4. Setup and Installation

### Prerequisites
-   Python 3.10+ (as per your environment, though 3.8+ from your old README is also likely fine)
-   Poetry for dependency management.

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url> # Replace with your actual repo URL
    cd <repository-name>
    ```
2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    This command creates a virtual environment (if one doesn't exist) and installs all packages specified in `pyproject.toml` and `poetry.lock`.

### Key Dependencies (from `pyproject.toml`)
-   `torch`: PyTorch core library.
-   `transformers`: Hugging Face Transformers library.
-   `datasets`: Hugging Face Datasets library.
-   `evaluate`: Hugging Face Evaluate library for metrics.
-   `tokenizers`: Hugging Face Tokenizers library.
-   `numpy`, `tqdm`, `pyyaml`, etc.

## 5. Dataset Details

-   **Default Dataset:** WMT14 French-English (`fr-en`), loaded via Hugging Face `datasets`.
-   **Configuration:** The dataset name (`--dataset_name`) and language pair/configuration (`--dataset_config_name`) can be specified via command-line arguments (see `DataTrainingArguments` in `train_revised.py`).
-   **Preprocessing Steps:**
    1.  **Tokenization:** Uses `MarianTokenizer` by default (configurable via `--tokenizer_name`, e.g., `Helsinki-NLP/opus-mt-fr-en`).
    2.  **Sequence Length Control:** Input and target sequences are truncated or padded to `max_source_length` and `max_target_length` respectively.
    3.  **Special Tokens:** Standard special tokens (`<pad>`, `<s>` (start-of-sequence, often implicitly handled or using `pad_token_id` for Marian), `</s>` (end-of-sequence)) are managed by the tokenizer.
-   **Data Caching:**
    -   **Raw Datasets:** Hugging Face `datasets` caches downloaded raw data in `~/.cache/huggingface/datasets/`.
    -   **Processed Datasets:** Tokenized and preprocessed datasets are automatically cached by the script in the `./data/processed/` directory within the project. This significantly speeds up subsequent runs. To force reprocessing, use the `--overwrite_cache` command-line argument.

## 6. Model Architecture

The model is a from-scratch PyTorch implementation of the Transformer architecture detailed in "Attention is All You Need":
-   **Embedding Layer:** Converts input token IDs (source and target) into dense vector representations (`dim_embedding`).
-   **Positional Encoding:** Adds sinusoidal positional information to the embeddings, allowing the model to understand token order as Transformers are inherently permutation-invariant.
-   **Encoder Stack:** Composed of `num_encoder_layers` identical `EncoderLayer` modules. Each `EncoderLayer` features:
    -   A **Multi-Head Self-Attention** sublayer: Allows each position in the input sequence to attend to all positions in the input sequence.
    -   A **Position-wise Feed-Forward Network (FFN)** sublayer: A fully connected feed-forward network applied independently to each position.
    -   Residual connections followed by Layer Normalization are applied around each of the two sublayers.
-   **Decoder Stack:** Composed of `num_decoder_layers` identical `DecoderLayer` modules. Each `DecoderLayer` features:
    -   A **Masked Multi-Head Self-Attention** sublayer: Similar to the encoder's self-attention, but masked to prevent positions from attending to subsequent positions (maintaining auto-regressive property for generation).
    -   A **Multi-Head Cross-Attention** sublayer: Allows each position in the decoder to attend to all positions in the output of the encoder stack. This is how information from the source sentence is incorporated.
    -   A **Position-wise Feed-Forward Network (FFN)** sublayer.
    -   Residual connections followed by Layer Normalization are applied around each of the three sublayers.
-   **Output Layer:** A final linear layer followed by a softmax function to produce a probability distribution over the target vocabulary for each output position.
-   **Loss Criterion:** The model is trained using Cross-Entropy Loss, comparing the predicted probability distribution with the actual target token.

## 7. Training Process

The main script for training and evaluation is `src/train_revised.py`.

### Running the Training Script
First, activate the Poetry virtual environment:
```bash
poetry shell
```
Then, execute the training script with desired arguments.

**Example (Quick Test Run - 5% data, 1 epoch):**
```bash
python src/train_revised.py \
  --output_dir ./outputs \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --num_train_epochs 1 \
  --dataset_fraction 0.05 \
  --max_source_length 128 \
  --max_target_length 128 \
  --logging_steps 50
```

**Example (More Substantial Training Run):**
```bash
python src/train_revised.py \
  --output_dir ./outputs \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 1000 \
  --save_strategy epoch \
  --logging_strategy steps \
  --logging_steps 100 \
  --save_total_limit 3 \
  --dataset_fraction 1.0 \
  --max_source_length 128 \
  --max_target_length 128 \
  --generation_num_beams 4
```
*(Note: The script attempts to handle MPS vs. CUDA for `fp16`. If `--fp16` is passed and MPS is detected, it will be disabled with a warning, as true `fp16` is for CUDA.)*

### Key Command-Line Arguments
Arguments are defined using dataclasses (`ModelArguments`, `DataTrainingArguments`, `Seq2SeqTrainingArguments`) in `train_revised.py`.

-   **General:**
    -   `--output_dir`: Directory for all outputs (checkpoints, logs). Default: `./outputs`.
    -   `--do_train`: Flag to enable training.
    -   `--do_eval`: Flag to enable evaluation.
-   **Dataset Related (`DataTrainingArguments`):**
    -   `--dataset_name`, `--dataset_config_name`: Specify dataset from Hugging Face Hub.
    -   `--dataset_fraction`: Proportion of dataset to use (0.0 to 1.0). Essential for debugging.
    -   `--max_source_length`, `--max_target_length`: Maximum sequence lengths for tokenization.
    -   `--tokenizer_name`: Identifier for the Hugging Face tokenizer (e.g., `Helsinki-NLP/opus-mt-fr-en`).
    -   `--overwrite_cache`: Force reprocessing of datasets.
-   **Model Architecture Related (`ModelArguments`):**
    -   `--dim_embedding`, `--num_heads`, `--num_encoder_layers`, `--num_decoder_layers`, `--dim_feedforward`, `--dropout`.
-   **Training Hyperparameters (`Seq2SeqTrainingArguments`):**
    -   `--per_device_train_batch_size`, `--per_device_eval_batch_size`.
    -   `--gradient_accumulation_steps`: Effective batch size = `batch_size * num_devices * grad_accum_steps`.
    -   `--learning_rate`, `--lr_scheduler_type`, `--warmup_steps`.
    -   `--num_train_epochs`.
    -   `--save_strategy`, `--logging_strategy`, `--logging_steps`, `--save_total_limit`.
    -   `--generation_num_beams`: For beam search during evaluation generation.
    -   Many others (refer to Hugging Face `Seq2SeqTrainingArguments` documentation).

### Resuming Training from a Checkpoint
The script is designed to automatically resume training:
1.  It checks for the existence of the `./outputs/checkpoint-last/` directory.
2.  If this directory contains a valid checkpoint (model weights, tokenizer files, `trainer_state.json`, `training_args.bin`), training will resume from that state.
3.  If not found, or if `--overwrite_output_dir` is specified, training starts from scratch.

To explicitly start fresh, either delete `./outputs/checkpoint-last/` or use `--overwrite_output_dir`.

### MPS (Mac M3 Pro) Specifics
-   The script auto-detects and defaults to MPS if available.
-   `fp16` (16-bit float) mixed precision: The script includes a check to disable `training_args.fp16` if MPS is the device, as `fp16` is primarily optimized for NVIDIA CUDA GPUs. MPS has its own path for lower-precision computation which might not align directly with the `fp16` flag in Hugging Face Trainer.

## 8. Evaluation

-   **Metrics Calculated:**
    -   **`eval_bleu` (BLEU Score):** The primary metric for assessing translation quality. Compares n-gram overlap between model outputs and reference translations.
    -   **`eval_bp` (Brevity Penalty):** A component of BLEU that penalizes translations significantly shorter than references.
    -   **`eval_gen_len` (Generated Length):** The average length of the translations produced by the model during evaluation.
    -   **`eval_loss` (Evaluation Loss):** The model's cross-entropy loss calculated on the evaluation dataset.
-   **Evaluation Process:**
    -   Triggered if `--do_eval` is active.
    -   Occurs at intervals defined by `training_args.evaluation_strategy` (programmatically set to "epoch" in the script to ensure end-of-epoch evaluation).
-   **Sample Translations:** For qualitative insight, the script decodes and prints a few examples from the evaluation set, showing the source text, the human reference translation, and the model's generated translation.

## 9. Inference (Using the Trained Model for Translation)

Once training is complete, the model can be used for translating new sentences.
1.  **Locating the Model:**
    -   The Hugging Face Trainer might save the "best" model based on a specified metric (if `load_best_model_at_end=True`) in a subdirectory like `outputs/checkpoint-<best_step>/`.
    -   The `./outputs/checkpoint-last/` directory contains the model, tokenizer, and trainer state at the absolute end of the last training run.
    -   The script also saves a specific inference-oriented checkpoint at `./outputs/optimized_model/model_optimized.pt`. This bundle includes the model's `state_dict`, vocabulary size, and the model's architectural configuration.

2.  **Loading and Using the "Optimized" Model:**
    The `train_revised.py` script contains a helper function `load_optimized_model` (around line 890) for this purpose. Here's a conceptual example of how to use it:

    ```python
    import torch
    from src.train_revised.py import load_optimized_model # Adjust import path as needed
    # Assuming your Transformer class and tokenizer are also accessible
    # from src.model.Transformer import Transformer (if not loaded by helper)
    # from transformers import MarianTokenizer (if not loaded by helper)

    # Configuration
    model_dir_path = "./outputs/optimized_model" # Path to the optimized model directory
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_optimized_model(model_dir_path, device=device)
    model.eval() # Set the model to evaluation mode (disables dropout, etc.)

    def translate_sentence(text_to_translate, model_instance, tokenizer_instance, device_instance, max_gen_length=128, num_beams_gen=4):
        """Translates a single sentence."""
        inputs = tokenizer_instance(text_to_translate, return_tensors="pt", max_length=512, truncation=True, padding=True)
        input_ids = inputs.input_ids.to(device_instance)
        attention_mask = inputs.attention_mask.to(device_instance) # Ensure your model.generate can use this
        
        with torch.no_grad(): # Disable gradient calculations for inference
            # Ensure your model's generate method uses pad_token_id or an appropriate BOS token
            generated_ids = model_instance.generate(
                input_ids,
                attention_mask=attention_mask, 
                max_length=max_gen_length,
                num_beams=num_beams_gen,
                decoder_start_token_id=tokenizer_instance.pad_token_id # For Marian-style models
                # early_stopping=True # Often a good idea for beam search
            )
        
        translated_text = tokenizer_instance.decode(generated_ids[0], skip_special_tokens=True)
        return translated_text

    # Example
    french_sentence = "Au Sénat, nous ne sommes que deux indépendants."
    english_translation = translate_sentence(french_sentence, model, tokenizer, device)
    print(f"Source (French): {french_sentence}")
    print(f"Model Translation (English): {english_translation}")
    ```
    *(Important: Ensure your custom `Transformer.generate` method is compatible with this inference pattern, particularly regarding the use of `attention_mask` and `decoder_start_token_id`.)*

## 10. Custom Components Explained

-   **`CustomDataCollator` (Defined in `src/train_revised.py` around line 159):**
    -   **Purpose:** This class is responsible for taking a list of individual tokenized examples from the dataset and forming a batch suitable for input to the Transformer model.
    -   **Key Functionality for Teacher Forcing:** Its primary role during training is to prepare `decoder_input_ids`. This is achieved by taking the `labels` (target token sequences), shifting them one position to the right, and prepending a start-of-sequence token (e.g., `tokenizer.pad_token_id` for Marian). This "teacher-forced" sequence is what the decoder uses as input at each step, being guided by the ground-truth previous tokens.
    -   **Other Responsibilities:**
        -   Handles dynamic padding of sequences within each batch to ensure all tensors have uniform dimensions.
        -   Converts lists of token IDs into PyTorch tensors.
        -   Moves all tensors to the active computation device (MPS, CUDA, or CPU).
        -   Prepares `labels` for loss calculation by typically replacing padding token IDs with -100 (which is ignored by PyTorch's CrossEntropyLoss).
-   **`TransformerTrainer` (Defined in `src/train_revised.py` around line 204):**
    -   **Purpose:** This class inherits from Hugging Face's `Seq2SeqTrainer` to allow for custom behavior within the training and evaluation loops.
    -   **Custom `prediction_step` Method:** The `prediction_step` is overridden. This method controls how the model makes predictions during evaluation (and potentially inference if `predict()` is called). Customizing it allows for:
        -   Handling specific output formats from your custom model.
        -   Implementing a tailored loss calculation for evaluation if it differs from training.
        -   Ensuring tensor shapes and sequence lengths are compatible, as seen in your implementation which carefully slices tensors before loss computation.

## 11. Troubleshooting & Key Learnings

-   **Argument Parsing:**
    -   The script uses `HfArgumentParser` with `ModelArguments`, `DataTrainingArguments`, and `Seq2SeqTrainingArguments`.
    -   `Seq2SeqTrainingArguments` (from Hugging Face) includes most standard training parameters like `evaluation_strategy`, `generation_num_beams`, etc. Custom arguments should be added to `ModelArguments` or `DataTrainingArguments`.
    -   If "unrecognized argument" errors occur, ensure the argument is defined in the correct dataclass or is a standard `Seq2SeqTrainingArguments` field. For boolean flags, use `--flag_name` or `--flag_name=True/False`. For others, `--arg_name=value` is often more robust than `--arg_name value`.
    -   The script now programmatically sets `training_args.evaluation_strategy = "epoch"` to ensure consistent behavior, bypassing potential CLI parsing issues for this specific argument.
-   **MPS and `fp16`:**
    -   True `fp16` mixed precision (activated by `--fp16` or `fp16=True` in `TrainingArguments`) is primarily designed for NVIDIA CUDA GPUs.
    -   The training script includes a check to disable `fp16` if MPS is detected, as MPS handles lower precision differently and may not support the standard `fp16` pathway of the Trainer, or it might lead to errors.
-   **Low BLEU Scores / Repetitive Output (Common with Small Data):**
    -   Training Transformer models for NMT from scratch is highly data-intensive. Initial experiments with small `dataset_fraction` values (e.g., 0.05) are excellent for debugging the pipeline but will likely result in poor translation quality (BLEU near 0, repetitive outputs).
    -   Meaningful translation performance requires training on a substantial portion (ideally 100%) of a large, high-quality parallel corpus for a sufficient number of epochs.
-   **Checkpoint Resumption:** The explicit saving mechanism for `./outputs/checkpoint-last/` (saving model, tokenizer, trainer state, and training args) has proven robust for ensuring reliable training resumption.

## 12. References

-   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762). *Advances in neural information processing systems, 30*.
-   Rush, A. (2018). [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). *Harvard NLP*.
-   [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
-   [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
-   [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)

This comprehensive README should serve as an excellent reference for yourself and your colleagues!
