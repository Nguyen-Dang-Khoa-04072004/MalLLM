from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
config = ModelConfig.from_json_file("config/deepseek-coder-6.7b.json")
prompt = Prompt.combine(
    Prompt("src/prompts/deepseek-system-prompt.txt",role="system"),
    Prompt("src/prompts/deepseek-user-prompt.txt")
)
model = AIModel(config)
print(model.generate(prompt))