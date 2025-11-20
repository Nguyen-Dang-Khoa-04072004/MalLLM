import argparse
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
from data.loader import Dataloader, DataSamples
from src.utils import response_parser
from typing import List

global checkpoint_set
checkpoint_set = set()

def load_checkpoints():
    global checkpoint_set
    output_path = Path("./output/checkpoints")
    if not output_path.exists():
        (output_path).mkdir(parents=True, exist_ok=True)
    
    file = output_path / "checkpoint.json"
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            checkpoint_set = set(data.get("processed_packages", []))
    else:
        checkpoint_set = set()

def save_checkpoints(package_name):
    global checkpoint_set
    output_path = Path("./output/checkpoints")
    file = output_path / "checkpoint.json"
    checkpoint_set.add(package_name)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump({"processed_packages": list(checkpoint_set)}, f, indent=4, ensure_ascii=False)
        

def process_chunk(chunk_samples: List[DataSamples], args, chunk_id, model:AIModel=None, system_prompt:Prompt=None) -> int:
    
    output_path = Path("./output")
    
    processed_count = 0
    
    for sample in tqdm(chunk_samples, desc=f"Worker-{chunk_id}", position=chunk_id):
        try:
            package_path = Path(sample.package_path)
            package_name = package_path.name.removesuffix('.txt')

            response_path = output_path / sample.label / (f"{package_name}.json")

            user_prompt = Prompt(sample.package_path, role="user")
            responses = model.generate(Prompt.combine(system_prompt, user_prompt))
            responses = response_parser.parse_llm_response(responses)
            
            (output_path / sample.label).mkdir(parents=True, exist_ok=True)
            
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)

            save_checkpoints(sample.package_name)

            processed_count += 1
            
        except Exception as e:
            print(f"[Worker-{chunk_id}] Error: {sample.package_path}: {e}")
    
    return processed_count

def main(parser_args):
    output_path = Path("./output")
    (output_path / "malicious").mkdir(parents=True, exist_ok=True)
    (output_path / "benign").mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    samples_to_process: List[DataSamples] = Dataloader(source_dir=parser_args.source_dir).load_data()

    global checkpoint_set
    load_checkpoints()

    samples_to_process = [
        s for s in samples_to_process if s.package_name not in checkpoint_set
    ]

    for file in (output_path / "malicious").glob("*.json") and (output_path / "benign").glob("*.json"):
        package_name = file.name.removesuffix('.json')
        if package_name in checkpoint_set:
            samples_to_process = [
                s for s in samples_to_process if s.package_name != package_name
            ]
            
    total_samples = len(samples_to_process)
    if total_samples == 0:
        print("Processing complete.")
        return

    num_workers = min(parser_args.max_workers, total_samples)
    
    chunk_size = math.ceil(total_samples / num_workers)
    chunks = [samples_to_process[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    model_config = ModelConfig.from_json_file(parser_args.config_file)
    model = AIModel(model_config)
    system_prompt = Prompt(parser_args.system_prompt, role="system")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, parser_args, i, model, system_prompt) for i, chunk in enumerate(chunks)]
        
        for i, future in enumerate(futures):
            try:
                count = future.result()
                print(f"Worker-{i} processed {count} samples")
            except Exception as e:
                print(f"Worker-{i} failed: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalLLM Code Analysis Tool")
    parser.add_argument("--config-file", type=str, default="config/deepseek-coder-6.7b.json", help="File path dẫn đến config file")
    parser.add_argument("--source-dir", type=str, default="./data/samples/")
    parser.add_argument("--system-prompt", type=str, default="src/prompts/deepseek-system-prompt.txt", help="File path đến system prompt")
    parser.add_argument("--max-workers", type=int, default=5, help="Số lượng process worker song song tối đa")
    
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main(parser.parse_args())