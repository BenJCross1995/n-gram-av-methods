# -*- coding: utf-8 -*-
"""
Model loading utilities.

Loads a local Hugging Face causal LM and its tokenizer with sensible defaults
for inference/scoring and optional multi-GPU device mapping.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_loc: str):
    """Load a local AutoModelForCausalLM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    model = AutoModelForCausalLM.from_pretrained(model_loc)
    model.eval()
    return tokenizer, model
