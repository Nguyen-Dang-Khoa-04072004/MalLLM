import argparse
from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
from data.loader import Dataloader
from src.utils import response_parser
import json
from pathlib import Path
from tqdm import tqdm

def main(parser_args):

    config = ModelConfig.from_json_file(parser_args.config_file)
    system_prompt = Prompt(parser_args.system_prompt, role="system")
    all_samples = Dataloader(source_dir=parser_args.source_dir).load_data()
    total_samples = len(all_samples)
    processed_samples = []
    model = AIModel(config)
    output_path = Path("./output")
    output_path.mkdir(parents=True, exist_ok=True)

    malware_path = output_path / "malicious"
    malware_path.mkdir(parents=True, exist_ok=True)

    benign_path = output_path / "benign"
    benign_path.mkdir(parents=True, exist_ok=True)

    # Checkpoint files
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Read checkpoint files if exist
    checkpoint_files = list(checkpoint_path.glob("*.json"))

    if checkpoint_files:
        processed_samples = []
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    processed_samples.append(item['file_path'])
        samples = [sample for sample in all_samples if sample.file_path not in processed_samples]
    else:
        samples = all_samples
    
    processed_count = len(processed_samples)
    
    pbar = tqdm(total=total_samples, initial=processed_count, desc="Processing samples", unit="sample")

    for sample in samples:

        user_prompt = Prompt(sample.file_path, role="user")
        responses = model.generate(Prompt.combine(
            system_prompt, user_prompt
        ))

        responses = response_parser.parse_llm_response(responses)
        file_path = Path(sample.file_path)
        file_name = f"{file_path.name.removesuffix('.txt')}.json"

        response_path = output_path / sample.label / file_name

        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)
        
        processed_samples.append(sample.file_path)
        
        checkpoint_file = checkpoint_path / "checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump([{"file_path": path} for path in processed_samples], f, indent=4, ensure_ascii=False)
        
        pbar.update(1)
    
    pbar.close()
        

        


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