import pandas as pd
import json
import jsonlines
from datasets import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import os
from config import DATA_DIR, DATASET_CONFIG

os.environ["HF_DATASETS_CACHE"] = "./hf_cache"


class CustomerSupportDataProcessor:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.config = DATASET_CONFIG

    def create_sample_data(self):
        """Create sample customer support data for demonstration"""
        sample_conversations = [
            {
                "category": "Account",
                "subcategory": "Login Issues",
                "customer_query": "I can't log into my account. It says my password is incorrect but I'm sure it's right.",
                "agent_response": "I understand how frustrating this can be. Let me help you resolve this login issue. First, please try resetting your password using the 'Forgot Password' link on the login page. If you continue to have trouble, I can assist you further with account verification."
            },
            {
                "category": "Billing",
                "subcategory": "Payment Issues",
                "customer_query": "My credit card was charged twice for the same order. Can you help me get a refund?",
                "agent_response": "I apologize for this billing error. I can see the duplicate charge on your account. I'll process a refund for the duplicate transaction right away. You should see the refund on your statement within 3-5 business days. Is there anything else I can help you with today?"
            },
            {
                "category": "Technical Support",
                "subcategory": "Software Issues",
                "customer_query": "The app keeps crashing when I try to upload photos. This is really annoying!",
                "agent_response": "I'm sorry to hear about the app crashes you're experiencing. This is definitely frustrating. Let's troubleshoot this together. First, please try closing and reopening the app. If that doesn't work, try restarting your device. If the issue persists, please let me know your device model and app version so I can provide more specific assistance."
            },
            {
                "category": "Product Information",
                "subcategory": "Features",
                "customer_query": "Does your premium plan include unlimited storage? I need to store a lot of files.",
                "agent_response": "Great question! Yes, our Premium plan does include unlimited cloud storage for all your files. You'll also get advanced sharing features, priority customer support, and enhanced security features. Would you like me to provide more details about upgrading to Premium or help you with the upgrade process?"
            },
            {
                "category": "Shipping",
                "subcategory": "Delivery",
                "customer_query": "My order was supposed to arrive yesterday but I still haven't received it. Where is my package?",
                "agent_response": "I understand your concern about the delayed delivery. Let me track your order right away. I can see that your package is currently in transit and should arrive within the next 24 hours. There was a slight delay due to weather conditions in your area. I'll send you a tracking link so you can monitor its progress. We apologize for any inconvenience this may have caused."
            },
            {
                "category": "Returns",
                "subcategory": "Return Process",
                "customer_query": "I want to return this item. It doesn't fit properly and I'm not satisfied with the quality.",
                "agent_response": "I'm sorry to hear that the item didn't meet your expectations. I'll be happy to help you with the return process. You have 30 days from the purchase date to return items. I'll email you a prepaid return label and detailed instructions. Once we receive the item, we'll process your refund within 5-7 business days. Is there anything specific about the sizing or quality that we should know about?"
            },
            {
                "category": "Account",
                "subcategory": "Profile Settings",
                "customer_query": "How do I change my email address on my account? I can't find the option anywhere.",
                "agent_response": "I'd be happy to help you update your email address. You can change it by going to Account Settings > Personal Information > Contact Details. Click on the edit button next to your email address and enter your new email. You'll need to verify the new email address before the change takes effect. If you're having trouble finding these options, I can walk you through it step by step."
            },
            {
                "category": "Technical Support",
                "subcategory": "Connectivity",
                "customer_query": "The website is loading very slowly on my computer. Is there a problem with your servers?",
                "agent_response": "Thank you for bringing this to our attention. I've checked our server status and everything appears to be running normally. The slow loading might be related to your internet connection or browser cache. Please try clearing your browser cache and cookies, or try accessing the site using a different browser. If you continue to experience slow loading, please let me know your location and internet provider so I can investigate further."
            }
        ]

        # Create JSONL format for training
        jsonl_data = []
        for conv in sample_conversations:
            # Format for instruction following
            instruction = f"Customer inquiry about {conv['category']} - {conv['subcategory']}: {conv['customer_query']}"
            response = conv['agent_response']

            jsonl_data.append({
                "instruction": instruction,
                "response": response,
                "category": conv['category'],
                "subcategory": conv['subcategory'],
                "input": ""  # Empty input for instruction-following format
            })

        # Save sample data
        sample_file = self.data_dir / "sample_data.jsonl"
        with open(sample_file, 'w') as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')

        print(f"Sample data created: {len(jsonl_data)} examples saved to {sample_file}")
        return jsonl_data

    def format_conversation_for_training(self, instruction, response):
        """Format a single conversation for training"""
        return {
            "text": self.config["instruction_template"].format(
                instruction=instruction,
                response=response
            )
        }

    def load_and_process_data(self, file_path=None, tokenizer=None):
        """Load and process data for training"""
        if file_path is None:
            file_path = self.data_dir / "customer_support_data.jsonl"

        # If main dataset doesn't exist, try sample data
        if not file_path.exists():
            file_path = self.data_dir / "sample_data.jsonl"

        # Create sample data if it doesn't exist
        if not file_path.exists():
            print("Sample data not found. Creating sample dataset...")
            self.create_sample_data()

        # Load data
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append(json.loads(line))
        except UnicodeDecodeError as e:
            print(f"Encoding error loading data: {e}")
            print("Trying with different encoding...")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
            except Exception as e2:
                print(f"Still failed with latin-1: {e2}")
                print("Creating sample data as fallback...")
                return self.create_sample_data(), None
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample data as fallback...")
            return self.create_sample_data(), None

        print(f"Loaded {len(data)} conversations from {file_path}")

        # Format for training
        formatted_data = []
        for item in data:
            formatted_item = self.format_conversation_for_training(
                item["instruction"],
                item["response"]
            )
            formatted_data.append(formatted_item)

        # Handle tokenization
        if tokenizer is not None:
            try:
                # Tokenize
                tokenized_data = tokenizer(
                    [item["text"] for item in formatted_data],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors=None
                )

                # Convert to HuggingFace dataset
                dataset = Dataset.from_dict(tokenized_data)

                # Split data
                split = dataset.train_test_split(
                    train_size=self.config["train_size"],
                    seed=42
                )

                train_dataset = split["train"]
                val_dataset = split["test"]

                print(f"Training set: {len(train_dataset)} examples")
                print(f"Validation set: {len(val_dataset)} examples")

                return train_dataset, val_dataset
            except Exception as e:
                print(f"Error during tokenization: {e}")
                print("Returning formatted data without tokenization")
                return formatted_data, None
        else:
            print("No tokenizer provided. Returning formatted data only.")
            return formatted_data, None

    def download_customer_support_dataset(self):
        try:
            import os
            import json
            from datetime import datetime
            from datasets import load_dataset

            os.environ["HF_DATASETS_CACHE"] = "./hf_cache"

            print("Downloading customer support dataset...")

            # Try different approaches to load the dataset
            try:
                # Method 1: Try loading with streaming first
                dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                                       split="train", streaming=True)

                # Convert streaming dataset to list
                dataset_list = list(dataset.take(10000))  # Take first 10k examples to avoid memory issues
                print(f"Successfully loaded {len(dataset_list)} examples via streaming")

            except Exception as stream_error:
                print(f"Streaming failed: {stream_error}")
                print("Trying regular download...")

                # Method 2: Regular download with error handling
                try:
                    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                                           split="train")
                    dataset_list = list(dataset)
                    print(f"Successfully loaded {len(dataset_list)} examples via regular download")

                except Exception as regular_error:
                    print(f"Regular download failed: {regular_error}")
                    print("Trying with trust_remote_code=True...")

                    # Method 3: Trust remote code
                    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                                           split="train", trust_remote_code=True)
                    dataset_list = list(dataset)
                    print(f"Successfully loaded {len(dataset_list)} examples with trust_remote_code")

            # Process the dataset
            processed_data = []
            skipped_count = 0

            for idx, item in enumerate(dataset_list):
                try:
                    instruction = item.get("instruction", "").strip()
                    response = item.get("response", "").strip()

                    if not instruction or not response:
                        skipped_count += 1
                        continue

                    processed_data.append({
                        "id": idx,
                        "instruction": instruction,
                        "response": response,
                        "category": item.get("category", "General"),
                        "input": item.get("input", "").strip(),
                        "metadata": {
                            "source": "bitext_customer_support",
                            "processed_at": datetime.now().isoformat(),
                            "original_index": idx
                        }
                    })

                    # Log progress every 1000 items
                    if idx % 1000 == 0 and idx > 0:
                        print(f"Processed {idx} items, {len(processed_data)} valid, {skipped_count} skipped")

                except Exception as item_error:
                    print(f"Error processing item {idx}: {item_error}")
                    skipped_count += 1
                    continue

            if len(processed_data) == 0:
                raise Exception("No valid data could be processed from the dataset")

            # Save with explicit UTF-8 encoding
            output_file = self.data_dir / "customer_support_data.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in processed_data:
                    json_str = json.dumps(item, ensure_ascii=False)
                    f.write(json_str + "\n")

            print(f"Real dataset downloaded: {len(processed_data)} examples saved to {output_file}")
            print(f"Skipped {skipped_count} invalid examples")
            return processed_data

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Using sample data instead...")
            return self.create_sample_data()