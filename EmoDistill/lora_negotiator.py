"""
LoRA-aware LLM wrapper for negotiation.

Drop-in compatible with `llm.llm_wrapper.LLMWrapper` interface, but loads:
  - Base model(e.g., Qwen2.5-7B-Instruct)on GPU
  - LoRA adapter(trained via lora_train.py)on top

Used in hierarchical eval where:
  - IQL policy picks emotion (meta level)
  - LoRA-fine-tuned LLM generates utterance (expression level)

Usage:
    wrapper = LoRAWrapper(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="results/lora/.../adapter_final",
        device="cuda",
    )
    response = wrapper.invoke([HumanMessage(content=prompt)], temperature=0.7)
    print(response.content)
"""

import os
from typing import List, Optional

import torch


class _MockMessage:
    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class LoRAWrapper:
    """Load base LLM + LoRA adapter, expose .invoke() like LLMWrapper."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        role: str = "creditor",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.role = role
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path

        print(f"📦 LoRAWrapper loading base: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Allow explicit cuda:N for 2-GPU split (tournament self-play needs creditor + debtor on different GPUs)
        if device.startswith("cuda:"):
            device_map = {"": device}
        elif device == "cuda":
            device_map = "auto"
        else:
            device_map = None
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
        }
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        if adapter_path:
            print(f"📦 LoRAWrapper loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(base, adapter_path)
            # Merge adapter into base for faster inference if not quantized.
            if not (load_in_8bit or load_in_4bit):
                self.model = self.model.merge_and_unload()
                print("   Merged adapter into base weights")
        else:
            self.model = base
            print("   No adapter (pure base model)")

        self.model.eval()
        if device == "cuda" and not (load_in_8bit or load_in_4bit):
            self.model = self.model.to("cuda")

    # -------- Public interface (matches LLMWrapper) --------

    def invoke(self, messages, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs):
        """Take langchain message list, return _MockMessage with .content."""
        prompt = self._messages_to_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        max_t = max_tokens if max_tokens is not None else self.max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_t,
                temperature=max(0.05, temperature),
                do_sample=temperature > 0.05,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )

        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        # Clean up Qwen-specific trailing tokens
        for end_token in ["<|im_end|>", "<|im_start|>", "</s>"]:
            if response.endswith(end_token):
                response = response[:-len(end_token)].strip()

        return _MockMessage(response)

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # -------- Helpers --------

    def _messages_to_prompt(self, messages) -> str:
        """Convert langchain messages to Qwen chat-template prompt."""
        chat_messages = []
        if isinstance(messages, str):
            chat_messages.append({"role": "user", "content": messages})
        else:
            for m in messages:
                if hasattr(m, "content"):
                    content = m.content
                    mtype = getattr(m, "type", "human")
                else:
                    content = str(m)
                    mtype = "human"
                if mtype == "ai":
                    role = "assistant"
                elif mtype == "system":
                    role = "system"
                else:
                    role = "user"
                chat_messages.append({"role": role, "content": content})

        return self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
