import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from pathlib import Path
from config import MODEL_CONFIG, INFERENCE_CONFIG, MODELS_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class CustomerSupportInference:
    def __init__(self, model_path=None):
        self.model_path = model_path or MODELS_DIR / "tinyllama-customer-support"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the fine-tuned model for inference"""
        print(f"Loading model from: {self.model_path}")

        if not self.model_path.exists():
            print("Fine-tuned model not found. Loading base model...")
            self.model_path = MODEL_CONFIG["base_model"]

        print("Loading model with transformers...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        print(f"Model loaded successfully on device: {self.model.device}")

    def format_prompt(self, customer_query, category=None):
        """Format customer query as a proper prompt"""
        if category:
            instruction = f"Customer inquiry about {category}: {customer_query}"
        else:
            instruction = f"Customer inquiry: {customer_query}"

        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def generate_response(self, customer_query, category=None, **kwargs):
        """Generate response to customer query"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Format the prompt
        prompt = self.format_prompt(customer_query, category)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device if needed
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generation parameters
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", INFERENCE_CONFIG["max_new_tokens"]),
            "temperature": kwargs.get("temperature", INFERENCE_CONFIG["temperature"]),
            "top_p": kwargs.get("top_p", INFERENCE_CONFIG["top_p"]),
            "top_k": kwargs.get("top_k", INFERENCE_CONFIG["top_k"]),
            "repetition_penalty": kwargs.get("repetition_penalty", INFERENCE_CONFIG["repetition_penalty"]),
            "do_sample": kwargs.get("do_sample", INFERENCE_CONFIG["do_sample"]),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Generate response
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )

        generation_time = time.time() - start_time

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new response (after the prompt)
        response = full_response[len(prompt):].strip()

        # Clean up response
        response = self._clean_response(response)

        return {
            "response": response,
            "generation_time": generation_time,
            "prompt_length": len(inputs["input_ids"][0]),
            "response_length": len(outputs[0]) - len(inputs["input_ids"][0])
        }

    def _clean_response(self, response):
        """Clean up the generated response"""
        # Remove common artifacts
        artifacts = ["### Instruction:", "### Response:", "<|endoftext|>", "<|end|>"]
        for artifact in artifacts:
            response = response.replace(artifact, "")

        # Remove extra whitespace
        response = response.strip()

        # Ensure response ends with proper punctuation
        if response and response[-1] not in ".!?":
            response += "."

        return response

    def batch_inference(self, queries):
        """Process multiple queries at once"""
        results = []

        for i, query in enumerate(queries):
            print(f"Processing query {i + 1}/{len(queries)}")

            if isinstance(query, dict):
                result = self.generate_response(
                    query.get("query", ""),
                    query.get("category", None)
                )
                result["query"] = query.get("query", "")
                result["category"] = query.get("category", "")
            else:
                result = self.generate_response(query)
                result["query"] = query
                result["category"] = ""

            results.append(result)

        return results

    def evaluate_model(self, test_queries=None):
        """Evaluate model performance on test queries"""
        if test_queries is None:
            test_queries = [
                {"query": "I can't log into my account", "category": "Account"},
                {"query": "My order hasn't arrived yet", "category": "Shipping"},
                {"query": "I was charged twice for the same item", "category": "Billing"},
                {"query": "The app keeps crashing on my phone", "category": "Technical Support"},
                {"query": "How do I cancel my subscription?", "category": "Account"},
            ]

        print("Evaluating model performance...")
        results = self.batch_inference(test_queries)

        # Calculate metrics
        total_time = sum(r["generation_time"] for r in results)
        avg_time = total_time / len(results)
        avg_response_length = sum(r["response_length"] for r in results) / len(results)

        print(f"\nEvaluation Results:")
        print(f"Total queries: {len(results)}")
        print(f"Average generation time: {avg_time:.2f} seconds")
        print(f"Average response length: {avg_response_length:.1f} tokens")
        print(f"Total time: {total_time:.2f} seconds")

        # Print sample responses
        print("\nSample Responses:")
        for i, result in enumerate(results[:3]):
            print(f"\n{i + 1}. Query: {result['query']}")
            print(f"   Category: {result['category']}")
            print(f"   Response: {result['response']}")
            print(f"   Time: {result['generation_time']:.2f}s")

        return results


def create_inference_demo():
    """Create a simple demo for testing the model"""
    print("=== Customer Support AI Demo ===")

    # Initialize inference engine
    inference = CustomerSupportInference()

    try:
        inference.load_model()

        print("\nModel loaded successfully!")
        print("Type 'quit' to exit, 'eval' to run evaluation\n")

        while True:
            query = input("Customer Query: ").strip()

            if query.lower() == 'quit':
                break
            elif query.lower() == 'eval':
                inference.evaluate_model()
                continue
            elif not query:
                continue

            # Get category (optional)
            category = input("Category (optional): ").strip()
            if not category:
                category = None

            # Generate response
            print("\nGenerating response...")
            result = inference.generate_response(query, category)

            print(f"\nAI Assistant: {result['response']}")
            print(f"(Generated in {result['generation_time']:.2f}s)")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run fine_tune_model.py first to train the model.")


def main():
    """Main inference script"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        create_inference_demo()
    else:
        # Run evaluation by default
        inference = CustomerSupportInference()
        inference.evaluate_model()


if __name__ == "__main__":
    main()