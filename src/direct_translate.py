#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct translation functions for the Transformer model.

This module provides simple translation functions for the custom Transformer model.
"""

import torch
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def translate_text(model, tokenizer, text, device):
    """
    Translate text using the model with greedy decoding.
    """
    # Tokenize input
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    try:
        with torch.no_grad():
            # Generate with simple greedy search
            outputs = model.generate(
                input_ids=input_ids,
                max_length=128,
                num_beams=1,  # Greedy decoding
                early_stopping=True
            )
            
            # Decode the outputs
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return "[Translation failed]"


def translate_batch(model, tokenizer, texts, device):
    """
    Translate a batch of texts.
    """
    translations = []
    for text in texts:
        translation = translate_text(model, tokenizer, text, device)
        translations.append(translation)
    return translations
