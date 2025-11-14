import config
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.model_config import ModelConfig
from src.prompts.prompt import Prompt
import torch

# Load model & tokenizer
# "deepseek-ai/deepseek-coder-6.7b-instruct"
class AIModel:
    def __init__(self, config : ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name,dtype=torch.float16, device_map="auto")
    def generate(self, prompt : Prompt) -> str:

        # Tokenize input properly with attention mask
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Ensure pad_token_id is set (some models don’t define one)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Generate output
        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=(inputs != self.tokenizer.pad_token_id),  # ✅ add attention mask
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            num_return_sequences=self.config.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode only generated tokens
        return self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
