"""Model loading and inference for LLM experiments."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any


class LLMInferenceEngine:
    """Wrapper for LLM inference with deterministic decoding."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        cache_dir: str = ".cache",
    ):
        """
        Initialize the LLM inference engine.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            dtype: Model dtype (float16, float32, bfloat16)
            cache_dir: Cache directory for model downloads
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float16)

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded successfully on {device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_strings: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt using deterministic decoding.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling parameter
            stop_strings: Optional list of stop strings

        Returns:
            Dictionary with 'text' (generated text) and 'num_tokens' (token count)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate with deterministic decoding (temperature=0 -> greedy)
        with torch.no_grad():
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt) :]  # Remove prompt
        num_tokens = outputs.shape[1] - input_length

        return {
            "text": generated_text.strip(),
            "num_tokens": num_tokens,
            "full_output": full_output,
        }

    def format_chat_prompt(
        self, user_message: str, system_message: str | None = None
    ) -> str:
        """
        Format a chat prompt using the model's chat template.

        Args:
            user_message: User's message
            system_message: Optional system message

        Returns:
            Formatted prompt string
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        # Use apply_chat_template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            if system_message:
                prompt = (
                    f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
                )
            else:
                prompt = f"User: {user_message}\n\nAssistant:"

        return prompt
