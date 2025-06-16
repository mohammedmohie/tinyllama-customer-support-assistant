import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model  # CHANGED: Removed prepare_model_for_kbit_training
import wandb
from data_processor import CustomerSupportDataProcessor
from config import (
    MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG,
    WANDB_CONFIG, MODELS_DIR, INFERENCE_CONFIG
)


class CustomerSupportFineTuner:
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        self.model = None
        self.tokenizer = None
        self.data_processor = CustomerSupportDataProcessor()

    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.use_wandb:
            wandb.init(
                project=WANDB_CONFIG["project"],
                entity=WANDB_CONFIG["entity"],
                name=WANDB_CONFIG["name"],
                tags=WANDB_CONFIG["tags"],
                config={
                    **MODEL_CONFIG,
                    **LORA_CONFIG,
                    **TRAINING_CONFIG
                }
            )

    def load_model(self):
        """Load TinyLlama model using standard transformers"""
        print("Loading TinyLlama 1.1B Chat model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["base_model"],
            trust_remote_code=True
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["base_model"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        print(f"Model loaded: {MODEL_CONFIG['base_model']}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")

    def setup_lora(self):
        """Configure LoRA for parameter-efficient fine-tuning"""
        print("Setting up LoRA configuration...")

        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def load_datasets(self):
        """Load and prepare training datasets"""
        print("Loading datasets...")

        # Initialize tokenizer if not already done
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            print("Initializing tokenizer...")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"],
                padding_side="right"
            )
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try to load real dataset first, fallback to sample data
        try:
            print("Attempting to download real dataset...")
            real_data = self.data_processor.download_real_dataset()
            train_dataset, val_dataset = self.data_processor.load_and_process_data(
                tokenizer=self.tokenizer
            )
        except Exception as e:
            print(f"Error loading real dataset: {e}")
            print("Using sample data...")
            train_dataset, val_dataset = self.data_processor.load_and_process_data(
                tokenizer=self.tokenizer
            )

        return train_dataset, val_dataset

    def create_training_arguments(self):
        """Create training arguments"""
        output_dir = MODELS_DIR / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        return TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            max_steps=TRAINING_CONFIG["max_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            fp16=TRAINING_CONFIG["fp16"] and torch.cuda.is_available(),
            logging_steps=TRAINING_CONFIG["logging_steps"],
            optim=TRAINING_CONFIG["optim"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            seed=TRAINING_CONFIG["seed"],
            save_strategy="steps",
            save_steps=20,
            evaluation_strategy="steps",
            eval_steps=20,
            logging_dir=str(output_dir / "logs"),
            report_to="wandb" if self.use_wandb else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Helps with Windows compatibility
        )

    def train(self):
        """Main training function"""
        print("Starting fine-tuning process...")

        # Setup
        self.setup_wandb()
        self.load_model()
        self.setup_lora()

        # Load data
        train_dataset, val_dataset = self.load_datasets()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.create_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Save the final model
        print("Training completed. Saving model...")
        self.save_model()

        # Close wandb
        if self.use_wandb:
            wandb.finish()

    def save_model(self):
        """Save the fine-tuned model"""
        save_path = MODELS_DIR / "tinyllama-customer-support"
        save_path.mkdir(exist_ok=True)

        # Save LoRA adapter
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        print(f"Model saved to: {save_path}")

        # Create model info file
        model_info = {
            "model_name": "TinyLlama Customer Support Assistant",
            "base_model": MODEL_CONFIG["base_model"],
            "training_config": TRAINING_CONFIG,
            "lora_config": LORA_CONFIG,
            "description": "Fine-tuned TinyLlama 1.1B Chat for customer support using LoRA",
            "usage": "Load with AutoModelForCausalLM.from_pretrained() and apply LoRA weights"
        }

        import json
        with open(save_path / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    def test_inference(self):
        """Test the trained model with sample queries"""
        print("Testing model inference...")

        self.model.eval()

        test_queries = [
            "Customer inquiry about Account - Login Issues: I forgot my password and need help resetting it.",
            "Customer inquiry about Billing - Payment Issues: Why was I charged twice for my subscription?",
            "Customer inquiry about Technical Support - Software Issues: The mobile app keeps freezing."
        ]

        for query in test_queries:
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=INFERENCE_CONFIG["max_new_tokens"],
                    temperature=INFERENCE_CONFIG["temperature"],
                    top_p=INFERENCE_CONFIG["top_p"],
                    do_sample=INFERENCE_CONFIG["do_sample"],
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nQuery: {query}")
            print(f"Response: {response[len(query):].strip()}")


def main():
    """Main training script"""
    print("=== TinyLlama Customer Support Fine-tuning ===")
    print("This script will fine-tune TinyLlama 1.1B Chat for customer support")
    print("using LoRA for efficient training.\n")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    # Initialize trainer
    use_wandb = input("Use Weights & Biases for logging? (y/n): ").lower() == 'y'

    trainer = CustomerSupportFineTuner(use_wandb=use_wandb)

    try:
        # Start training
        trainer.train()

        # Test the model
        print("\n" + "=" * 50)
        test_model = input("Test the trained model? (y/n): ").lower() == 'y'
        if test_model:
            trainer.test_inference()

        print("\nFine-tuning completed successfully!")
        print(f"Model saved in: {MODELS_DIR}/tinyllama-customer-support")

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()