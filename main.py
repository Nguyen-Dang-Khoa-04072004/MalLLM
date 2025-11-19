import argparse
from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
from data.loader import Dataloader
from src.utils import response_parser
import json
from pathlib import Path

def main(parser_args):

    config = ModelConfig.from_json_file(parser_args.config_file)
    system_prompt = Prompt(parser_args.system_prompt, role="system")
    samples = Dataloader(source_dir=parser_args.source_dir).load_data()
    model = AIModel(config)
    output_path = Path("./output")
    output_path.mkdir(parents=True, exist_ok=True)

    malware_path = output_path / "malicious"
    malware_path.mkdir(parents=True, exist_ok=True)

    benign_path = output_path / "benign"
    benign_path.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        user_prompt = Prompt(sample.file_path, role="user")
        responses = model.generate(Prompt.combine(
            system_prompt, user_prompt
        ))

        # print(response_parser.parse_llm_response(responses))
        responses = response_parser.parse_llm_response(responses)
        file_path = Path(sample.file_path)
        file_name = f"{file_path.name.removesuffix(".txt")}.json"

        response_path = output_path / sample.label / file_name

        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)

        


if __name__ == "__main__":

    """
    Main application entry point

    Parameters
    ----------
        --config-file : str
            Path to the model configuration JSON file.

        --source-dir : str
            Path to the source directory containing data samples.

        --system-prompt : str
            Path to the system prompt text file.
    """

    parser = argparse.ArgumentParser(description="Run AI Model with specified configuration.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/deepseek-coder-6.7b.json",
        help="Path to the model configuration JSON file.",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="./data/samples/",
        help="Path to the source directory containing data samples.",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default="src/prompts/deepseek-system-prompt.txt",
    )

    main(parser.parse_args())