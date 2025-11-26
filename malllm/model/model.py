from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model_config import ModelConfig
from .prompt import Prompt
import torch

# Load model & tokenizer
# "deepseek-ai/deepseek-coder-6.7b-instruct"
class AIModel:
    def __init__(self, config : ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name,dtype=torch.float16, device_map="auto")
    def read_file(self, file_path : Path):
        # Nếu file_path là relative, resolve dựa trên folder model.py
        if not file_path.exists() and not file_path.is_absolute():
            file_path = Path(__file__).parent / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def tokenize(self, input_file_path : Path): 
        # Read prompts from given file
        prompt = [
            {"role": "system", "content": self.read_file(Path("system-prompt.txt"))},
            {"role": "system", "content": self.read_file(input_file_path)}
        ]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return inputs
    def generate(self, inputs) -> str:
        # Generate output
        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=(inputs != self.tokenizer.pad_token_id),
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
        
    def generate_batch(self, prompts : list[Prompt]) -> list[str]:
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            for prompt in prompts
        ]

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True
        ).to(self.model.device)

        with torch.no_grad(), torch.autocast(device_type=self.model.device.type, dtype=torch.float16):
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                num_return_sequences=1, # Batching thì nên chỉ trả về 1 sequence mỗi prompt
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]

        generated_tokens = outputs[:, input_length:]

        decoded_responses = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )

        return decoded_responses