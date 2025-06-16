"""
Configuration file for Customer Support AI Assistant
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Model configuration
MODEL_CONFIG = {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_seq_length": 2048,
    "dtype": "float16",
    "load_in_4bit": False,
    "use_gradient_checkpointing": True,
    "random_state": 42,
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "use_gradient_checkpointing": True,
    "random_state": 42,
    "use_rslora": False,
    "loftq_config": None,
}

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 9000,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 1,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
}

# Dataset configuration
DATASET_CONFIG = {
    "train_size": 0.9,
    "validation_size": 0.1,
    "max_length": 512,
    "instruction_template": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    "system_message": "You are a helpful customer support assistant. Provide clear, professional, and empathetic responses to customer inquiries.",
}

# Flask app configuration
FLASK_CONFIG = {
    "SECRET_KEY": "your-secret-key-here",
    "DEBUG": True,
    "HOST": "0.0.0.0",
    "PORT": 5000,
}

# Inference configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# Wandb configuration
WANDB_CONFIG = {
    "project": "customer-support-ai",
    "entity": None,
    "name": "tinyllama-customer-support",
    "tags": ["tinyllama", "customer-support", "fine-tuning"],
}

# Data sources
DATA_SOURCES = {
    "customer_support_dataset": {
        "url": "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "local_path": DATA_DIR / "customer_support_data.csv",
        "description": "Customer support conversation dataset for fine-tuning"
    },
    "sample_dataset": {
        "local_path": DATA_DIR / "sample_data.jsonl",
        "description": "Sample customer support conversations in JSONL format"
    }
}
