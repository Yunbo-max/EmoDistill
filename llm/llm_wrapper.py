"""
LLM wrapper for online/offline baselines
"""

import os
from typing import Optional, Union, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from openai import OpenAI

# Try to import offline baselines
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

class LLMWrapper:
    """Wrapper for LLM baselines (online and offline)"""
    
    def __init__(self, model_name: str, role: str = "generic", 
                 quantize: str = None, max_memory_gb: float = None, lazy_load: bool = False):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: Model name
            role: Role for logging
            quantize: None, "8bit", "4bit" for memory optimization
            max_memory_gb: Maximum GPU memory to use (e.g., 20 for 24GB card)
            lazy_load: If True, don't load model until first use
        """
        self.model_name = model_name
        self.role = role
        self.quantize = quantize
        self.max_memory_gb = max_memory_gb
        self.lazy_load = lazy_load
        self.model = self._initialize_model(model_name)
    
    # DashScope (Aliyun) API model names — must be checked BEFORE the offline-qwen
    # branch, otherwise "qwen-plus" would be misrouted to HuggingFace loading.
    DASHSCOPE_API_MODELS = {
        # Qwen API products (production)
        "qwen-plus", "qwen-plus-latest", "qwen-plus-2025-04-28",
        "qwen-turbo", "qwen-max", "qwen-max-latest", "qwen-max-longcontext",
        # Qwen 3.5 product line
        "qwen3.5-plus",
        # Qwen2 / Qwen2.5 (HF-style names supported by DashScope)
        "qwen2-7b-instruct", "qwen2-72b-instruct",
        "qwen2.5-7b-instruct", "qwen2.5-14b-instruct",
        "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",
        # Qwen3 series
        "qwen3-8b", "qwen3-14b", "qwen3-30b-a3b", "qwen3-32b",
        "qwen3-32b-instruct", "qwen3-235b-a22b",
        # Reasoning variants
        "qwq-32b", "qwq-32b-preview",
        # DeepSeek baselines hosted by DashScope
        "deepseek-v3", "deepseek-r1", "deepseek-r1-distill-qwen-7b", "deepseek-r1-distill-qwen-14b",
        # VL
        "qwen-vl-plus", "qwen-vl-max",
    }

    def _initialize_model(self, model_name: str):
        """Initialize LLM model"""
        model_lower = model_name.lower()

        # DashScope (Aliyun) API — explicit prefix OR recognized API model name.
        if model_lower.startswith("dashscope:") or model_lower in self.DASHSCOPE_API_MODELS:
            return self._initialize_dashscope(model_name)

        # DeepSeek API baselines (check FIRST before offline deepseek check)
        if "deepseek-chat" in model_lower or "deepseek-api" in model_lower:
            return self._initialize_deepseek_api()
        
        # Check for offline baselines
        elif "qwen" in model_lower or "deepseek" in model_lower or "llama" in model_lower or "offline:" in model_lower:
            if not TRANSFORMERS_AVAILABLE:
                print(f"⚠️ Transformers not available for {self.role}. Falling back to GPT-4o-mini")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            # Remove "offline:" prefix
            actual_model = model_name.replace("offline:", "")
            
            # Handle common model mappings for Qwen
            if "qwen" in actual_model.lower():
                qwen_model_map = {
                    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", 
                    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",  # Alternative format
                    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
                    "qwen-4b": "Qwen/Qwen2.5-3B-Instruct",  # Use 3B as closest to 4B
                    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
                    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
                    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
                    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
                    "qwen": "Qwen/Qwen2.5-7B-Instruct"  # Default to 7B
                }
                
                # Find the best match (check exact matches first, then partial)
                model_key = None
                # First check for exact matches
                for key in qwen_model_map.keys():
                    if actual_model.lower() == key or actual_model.lower() == f"offline:{key}":
                        model_key = key
                        break
                
                # If no exact match, check for partial matches
                if not model_key:
                    for key in qwen_model_map.keys():
                        if key in actual_model.lower():
                            model_key = key
                            break
                
                if model_key:
                    actual_model = qwen_model_map[model_key]
                    print(f"📋 Mapped {model_name} -> {actual_model}")
                else:
                    actual_model = "Qwen/Qwen2.5-7B-Instruct"  # Default fallback
                    
            # Handle common model mappings for DeepSeek
            elif "deepseek" in actual_model.lower():
                deepseek_model_map = {
                    "deepseek-1.3b": "deepseek-ai/deepseek-llm-1.3b-chat",
                    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
                    "deepseek-67b": "deepseek-ai/deepseek-llm-67b-chat",
                    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
                    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
                    "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
                    "deepseek": "deepseek-ai/deepseek-llm-7b-chat"  # Default to 7B
                }
                
                # Find the best match
                model_key = None
                for key in deepseek_model_map.keys():
                    if key in actual_model.lower():
                        model_key = key
                        break
                
                if model_key:
                    actual_model = deepseek_model_map[model_key]
                    print(f"📋 Mapped {model_name} -> {actual_model}")
                else:
                    actual_model = "deepseek-ai/deepseek-llm-7b-chat"  # Default fallback
            elif actual_model.lower() == "llama-7b" or actual_model.lower() == "llama":
                actual_model = "meta-llama/Llama-2-7b-chat-hf"
            
            try:
                print(f"🔄 Initializing offline model for {self.role}: {actual_model}")
                # Check if model exists on HuggingFace Hub
                from huggingface_hub import model_info
                try:
                    model_info(actual_model)
                    print(f"✅ Model {actual_model} found on HuggingFace Hub")
                except Exception as hub_e:
                    print(f"⚠️ Model {actual_model} not found on Hub: {hub_e}")
                    print(f"🔄 Trying to load anyway (might be cached locally)...")
                
                return self._initialize_offline_model(actual_model)
            except ImportError as ie:
                print(f"❌ Missing dependencies for offline model: {ie}")
                print("💡 Install with: pip install transformers torch accelerate")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            except Exception as e:
                print(f"❌ Failed to load offline model for {self.role}: {e}")
                print(f"🔄 Falling back to GPT-4o-mini...")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Online baselines
        elif "gpt" in model_lower:
            temp = 1.0 if "gpt-5" in model_lower else 0.7
            return ChatOpenAI(model=model_name, temperature=temp)
        elif "claude" in model_lower:
            return ChatAnthropic(model=model_name, temperature=0.7)
        else:
            # Default fallback
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def _initialize_offline_model(self, model_name: str):
        """Initialize offline model using transformers"""
        
        # Check if it's a Qwen or DeepSeek model
        is_qwen = "qwen" in model_name.lower()
        is_deepseek = "deepseek" in model_name.lower()
        
        class OfflineLLM:
            def __init__(self, model_name, parent_wrapper=None):
                self.model_name = model_name
                self.is_qwen = is_qwen
                self.is_deepseek = is_deepseek
                self.parent_wrapper = parent_wrapper or self
                self.model = None  # For lazy loading
                self.tokenizer = None  # For lazy loading
                
                # Only load tokenizer initially (lightweight)
                self._load_tokenizer()
                
                # Don't load model immediately - wait for first use or explicit load
                print(f"📋 Model wrapper initialized for {model_name} (model will load on first use)")
            
            def _format_prompt_for_model(self, messages):
                """Format prompt according to model's expected format"""
                if isinstance(messages, list) and len(messages) > 0:
                    if hasattr(messages[0], 'content'):
                        prompt = messages[0].content
                    else:
                        prompt = str(messages[0])
                else:
                    prompt = str(messages)
                
                if self.is_qwen:
                    # Qwen typically uses chat format with <|im_start|> and <|im_end|> tokens
                    # Check if tokenizer has chat template
                    if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                        try:
                            # Format using the model's chat template
                            formatted = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": prompt}],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            return formatted
                        except:
                            # Fallback format for Qwen
                            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        # Simple Qwen format
                        return f"Human: {prompt}\n\nAssistant:"
                elif self.is_deepseek:
                    # DeepSeek format
                    # Check if tokenizer has chat template
                    if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                        try:
                            # Format using the model's chat template
                            formatted = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": prompt}],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            return formatted
                        except:
                            # Fallback format for DeepSeek
                            return f"User: {prompt}\n\nAssistant:"
                    else:
                        # Simple DeepSeek format
                        return f"User: {prompt}\n\nAssistant:"
                else:
                    # Default format for other baselines
                    return f"User: {prompt}\n\nAssistant:"
            
            def invoke(self, messages, temperature=0.7, max_new_tokens=512):
                # Ensure model is loaded (reload if needed after cleanup)
                self.ensure_model_loaded()
                
                # Format prompt
                formatted_prompt = self._format_prompt_for_model(messages)
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                # Move to device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=max(temperature, 0.1),  # Ensure temperature > 0
                        do_sample=True if temperature > 0.1 else False,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1  # Reduce repetition
                    )
                
                # Decode response (skip the input/prompt part)
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Clean up response for Qwen and DeepSeek baselines
                if self.is_qwen:
                    # Remove any remaining special tokens
                    response = response.replace('<|im_end|>', '').replace('<|im_start|>', '')
                    # Remove any trailing assistant tag
                    if response.endswith('assistant'):
                        response = response[:-len('assistant')].strip()
                elif self.is_deepseek:
                    # Clean up DeepSeek specific tokens if needed
                    response = response.strip()
                    # Remove any common trailing patterns
                    if response.endswith('Assistant:'):
                        response = response[:-len('Assistant:')].strip()
                
                # Return compatible format
                class MockAIMessage:
                    def __init__(self, content):
                        self.content = content
                
                return MockAIMessage(response)
            
            def load_model_if_needed(self):
                """Load model only when needed (for lazy loading)"""
                if self.model is None:
                    print(f"🔄 Loading model on-demand: {self.model_name}")
                    self._load_model()
            
            def unload_model(self):
                """Unload model to free GPU memory"""
                if self.model is not None:
                    print(f"🗑️ Unloading model to free memory: {self.model_name}")
                    del self.model
                    self.model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"🧹 GPU cache cleared")
            
            def _load_model_and_tokenizer(self):
                """Internal method to load model and tokenizer"""
                # Load tokenizer first (lightweight)
                if self.tokenizer is None:
                    self._load_tokenizer()
                
                # Then load the heavy model
                self._load_model()
            
            def _load_tokenizer(self):
                """Load tokenizer (moved from __init__ for lazy loading)"""
                if self.is_qwen:
                    # Qwen baselines typically use special tokens
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True,
                        padding_side='left'  # Important for generation
                    )
                elif self.is_deepseek:
                    # DeepSeek baselines settings
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True,
                        padding_side='left'
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True
                    )
                
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def _load_model(self):
                """Load the actual model (tokenizer should already be loaded)"""
                if self.model is not None:
                    print(f"⚠️ Model {self.model_name} already loaded, skipping...")
                    return
                    
                # Determine device and optimize loading
                if torch.cuda.is_available():
                    print(f"🚀 Loading {self.model_name} on CUDA...")
                    device_map = "auto"  # Let transformers handle device mapping
                    torch_dtype = torch.float16
                    
                    # Apply quantization based on parent's settings
                    load_in_8bit = self.parent_wrapper.quantize == "8bit"
                    load_in_4bit = self.parent_wrapper.quantize == "4bit"
                    
                    if load_in_8bit:
                        print(f"🔧 Using 8-bit quantization for {self.model_name}")
                    elif load_in_4bit:
                        print(f"🔧 Using 4-bit quantization for {self.model_name}")
                        
                    # Set max memory per GPU if specified
                    max_memory = None
                    if self.parent_wrapper.max_memory_gb:
                        max_memory = {0: f"{self.parent_wrapper.max_memory_gb}GB"}
                        print(f"🎚️ Limiting GPU memory to {self.parent_wrapper.max_memory_gb}GB")
                        
                else:
                    print(f"💻 Loading {self.model_name} on CPU...")
                    device_map = "cpu"
                    torch_dtype = torch.float32
                    load_in_8bit = False
                    load_in_4bit = False
                    max_memory = None
                
                # Load model with optimizations
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,  # Optimize CPU memory usage
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                }
                
                if max_memory:
                    model_kwargs["max_memory"] = max_memory
                
                # Add quantization dependencies if needed
                if load_in_8bit or load_in_4bit:
                    try:
                        import bitsandbytes
                    except ImportError:
                        print("⚠️ bitsandbytes not installed. Install with: pip install bitsandbytes")
                        load_in_8bit = False
                        load_in_4bit = False
                        model_kwargs["load_in_8bit"] = False
                        model_kwargs["load_in_4bit"] = False
                
                print(f"📦 Loading model with config: {model_kwargs}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Model-specific setup
                if self.is_qwen and hasattr(self.tokenizer, 'chat_template'):
                    print(f"✅ Using Qwen chat template for {self.model_name}")
                elif self.is_deepseek and hasattr(self.tokenizer, 'chat_template'):
                    print(f"✅ Using DeepSeek chat template for {self.model_name}")
                
                print(f"✅ Model {self.model_name} loaded successfully!")
            
            def cleanup(self):
                """Clean up GPU memory by temporarily unloading model"""
                try:
                    if self.model and torch.cuda.is_available():
                        print(f"🧹 Freeing GPU memory for {self.model_name}...")
                        
                        # Method 1: Delete model from GPU and clear cache
                        try:
                            # Store model configuration for potential reloading
                            model_config = {
                                'model_name': self.model_name,
                                'device_info': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
                            }
                            
                            # Delete the model to free GPU memory
                            del self.model
                            self.model = None
                            
                            # Clear GPU cache
                            torch.cuda.empty_cache()
                            
                            # Additional aggressive cleanup
                            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                                torch.cuda.reset_accumulated_memory_stats()
                            if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                                torch.cuda.reset_max_memory_allocated()
                            
                            print(f"✅ GPU memory freed for {model_config['model_name']}")
                            
                        except Exception as e:
                            print(f"⚠️ Standard cleanup failed for {self.model_name}: {e}")
                            # Fallback - just clear cache
                            torch.cuda.empty_cache()
                            print(f"🧹 Fallback cleanup completed for {self.model_name}")
                            
                except Exception as e:
                    print(f"⚠️ All cleanup failed for {self.model_name}: {e}")
            
            def ensure_model_loaded(self):
                """Ensure model is loaded on GPU (reload if needed after cleanup)"""
                if self.model is None:
                    print(f"🔄 Reloading model after cleanup: {self.model_name}")
                    self._load_model()
            
            def __del__(self):
                """Destructor to ensure cleanup"""
                try:
                    self.cleanup()
                except:
                    pass  # Ignore errors during destruction
        
        offline_model = OfflineLLM(model_name, self)
        
        # Load immediately only if not using lazy loading
        if not self.lazy_load:
            print(f"🔄 Loading model immediately (lazy_load=False)")
            offline_model._load_model()
        else:
            print(f"⏳ Lazy loading enabled - model will load on first use")
        
        return offline_model
    
    def _initialize_dashscope(self, model_name: str):
        """Initialize DashScope (Aliyun Qwen) API via the round-robin wrapper."""
        from EmoDistill.dashscope_wrapper import DashScopeWrapper

        actual = model_name
        if actual.lower().startswith("dashscope:"):
            actual = actual.split(":", 1)[1]
        print(f"✅ DashScope client initialized: {actual} (round-robin across keys)")
        return DashScopeWrapper(model=actual, role=self.role)

    def _initialize_deepseek_api(self):
        """Initialize DeepSeek API client using OpenAI format"""
        
        # Get API key from environment
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            print("⚠️ DEEPSEEK_API_KEY not found in environment variables")
            print("🔄 Falling back to GPT-4o-mini...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        class DeepSeekAPI:
            """DeepSeek API wrapper using OpenAI client format"""
            
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                print(f"✅ DeepSeek API client initialized")
            
            def invoke(self, messages, temperature: float = 0.7, max_tokens: int = 512):
                """Invoke DeepSeek API with OpenAI-compatible format"""
                
                # Convert LangChain messages to OpenAI format
                api_messages = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        content = msg.content
                    else:
                        content = str(msg)
                    
                    # Determine role based on message type
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            role = 'user'
                        elif msg.type == 'ai':
                            role = 'assistant'
                        elif msg.type == 'system':
                            role = 'system'
                        else:
                            role = 'user'  # Default
                    else:
                        role = 'user'  # Default for string messages
                    
                    api_messages.append({
                        "role": role,
                        "content": content
                    })
                
                # If no system message, add a generic one
                if not any(msg["role"] == "system" for msg in api_messages):
                    api_messages.insert(0, {
                        "role": "system", 
                        "content": "You are a helpful assistant specializing in negotiation and communication."
                    })
                
                try:
                    # Call DeepSeek API
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    
                    # Extract response content
                    content = response.choices[0].message.content
                    
                    # Return in LangChain-compatible format
                    class MockAIMessage:
                        def __init__(self, content: str):
                            self.content = content
                            self.type = 'ai'
                    
                    return MockAIMessage(content)
                    
                except Exception as e:
                    print(f"❌ DeepSeek API error: {e}")
                    # Return error message
                    class MockAIMessage:
                        def __init__(self, content: str):
                            self.content = content
                            self.type = 'ai'
                    
                    return MockAIMessage(f"Error: DeepSeek API call failed - {str(e)}")
            
            def cleanup(self):
                """No cleanup needed for API calls"""
                pass
        
        return DeepSeekAPI(api_key)
    
    def invoke(self, messages, temperature: float = 0.7, **kwargs):
        """Invoke the LLM model"""
        return self.model.invoke(messages, temperature=temperature, **kwargs)
    
    def cleanup(self):
        """Clean up GPU memory (for offline baselines)"""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        elif hasattr(self.model, 'unload_model'):
            self.model.unload_model()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
            # Move offline model to CPU to free GPU memory
            try:
                if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                    print(f"🧹 Cleaning up GPU memory for {self.role}...")
                    self.model.model.to('cpu')
                    torch.cuda.empty_cache()
                    print(f"✅ GPU memory cleaned for {self.role}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup GPU memory for {self.role}: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    def cleanup(self):
        """Clean up GPU memory (for offline baselines)"""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
            # Move offline model to CPU to free GPU memory
            try:
                if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                    print(f"🧹 Cleaning up GPU memory for {self.role}...")
                    self.model.model.to('cpu')
                    torch.cuda.empty_cache()
                    print(f"✅ GPU memory cleaned for {self.role}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup GPU memory for {self.role}: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    def unload_model(self):
        """Unload model to free GPU memory"""
        if hasattr(self.model, 'unload_model'):
            self.model.unload_model()
    
    def reload_model(self):
        """Reload model (useful after unloading)"""
        if hasattr(self.model, '_load_model_and_tokenizer'):
            self.model._load_model_and_tokenizer()
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'cached_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'message': 'CUDA not available'}


# Alternative: Specialized Qwen wrapper
class QwenLLMWrapper(LLMWrapper):
    """Specialized wrapper for Qwen baselines"""
    
    def __init__(self, model_size: str = "7b", role: str = "generic"):
        """
        Initialize Qwen model wrapper
        
        Args:
            model_size: "3b", "7b", "14b", "72b" etc.
            role: Role description for logging
        """
        # Map model sizes to HuggingFace model names
        model_map = {
            "3b": "Qwen/Qwen2.5-3B-Instruct",
            "7b": "Qwen/Qwen2.5-7B-Instruct",
            "14b": "Qwen/Qwen2.5-14B-Instruct",
            "72b": "Qwen/Qwen2.5-72B-Instruct",
            "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            "0.5b": "Qwen/Qwen2.5-0.5B-Instruct"
        }
        
        model_name = model_map.get(model_size.lower(), "Qwen/Qwen2.5-7B-Instruct")
        super().__init__(f"offline:{model_name}", role)


# Alternative: Specialized DeepSeek wrapper with memory optimization
class DeepSeekLLMWrapper(LLMWrapper):
    """Specialized wrapper for DeepSeek baselines with 24GB VRAM optimization"""
    
    def __init__(self, model_size: str = "7b", role: str = "generic", 
                 quantize: str = "8bit", max_memory_gb: float = 20):
        """
        Initialize DeepSeek model wrapper optimized for 24GB VRAM
        
        Args:
            model_size: "1.3b", "7b", "67b" for chat baselines, or "coder-1.3b", "coder-6.7b", "coder-33b" for coder baselines
            role: Role description for logging
            quantize: "8bit" (recommended), "4bit" (max saving), None (standard)
            max_memory_gb: Max GPU memory (default 20GB for 24GB card safety)
        """
        # Map model sizes to HuggingFace model names
        model_map = {
            "1.3b": "deepseek-ai/deepseek-llm-1.3b-chat",
            "7b": "deepseek-ai/deepseek-llm-7b-chat",
            "67b": "deepseek-ai/deepseek-llm-67b-chat",
            "coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "coder-33b": "deepseek-ai/deepseek-coder-33b-instruct"
        }
        
        model_name = model_map.get(model_size.lower(), "deepseek-ai/deepseek-llm-7b-chat")
        print(f"🎯 Initializing DeepSeek-{model_size} with {quantize or 'fp16'} quantization")
        print(f"📊 Memory limit: {max_memory_gb}GB")
        
        super().__init__(f"offline:{model_name}", role, quantize, max_memory_gb)


# Example usage:
if __name__ == "__main__":
    # Initialize different baselines
    models_to_test = [
        ("qwen-7b", QwenLLMWrapper("7b", "negotiator")),
        ("qwen-3b", QwenLLMWrapper("3b", "negotiator")),
        ("deepseek-7b", DeepSeekLLMWrapper("7b", "negotiator")),
        ("deepseek-1.3b", DeepSeekLLMWrapper("1.3b", "negotiator")),
        ("deepseek-coder-6.7b", DeepSeekLLMWrapper("coder-6.7b", "negotiator")),
        ("deepseek-generic", LLMWrapper("offline:deepseek-7b", "negotiator")),
        ("deepseek-api", LLMWrapper("deepseek-chat", "negotiator")),  # New DeepSeek API
        ("llama", LLMWrapper("offline:llama-7b", "negotiator")),
        ("gpt-4", LLMWrapper("gpt-4o", "negotiator"))
    ]
    
    # Test prompt
    test_prompt = "You are negotiating a business deal. The other party offers $100. What's your counter offer?"
    
    for model_name, model in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}...")
        print(f"{'='*50}")
        
        try:
            response = model.invoke([HumanMessage(content=test_prompt)])
            print(f"Response from {model_name}: {response.content[:200]}...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")