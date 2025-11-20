import argparse
import json
import math
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
from data.loader import Dataloader
from src.utils import response_parser

def load_checkpoint(checkpoint_file):
    max_retries = 5
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return {item['file_path'] for item in data}, data
            return set(), []
        except (json.JSONDecodeError, IOError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return set(), []

def save_checkpoint(checkpoint_file, processed_list):
    max_retries = 5
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(processed_list, f, indent=4, ensure_ascii=False)
            temp_file.replace(checkpoint_file)
            return True
        except IOError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False

def process_chunk(chunk_samples, args, chunk_id, force):
    config = ModelConfig.from_json_file(args.config_file)
    model = AIModel(config)
    system_prompt = Prompt(args.system_prompt, role="system")
    checkpoint_dir = Path("./output/checkpoints")
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    output_path = Path("./output")
    
    processed_count = 0
    
    for sample in tqdm(chunk_samples, desc=f"Worker-{chunk_id}", position=chunk_id):
        try:
            file_path = Path(sample.file_path)
            file_name = f"{file_path.name.removesuffix('.txt')}.json"
            response_path = output_path / sample.label / file_name
            
            processed_set, processed_list = load_checkpoint(checkpoint_file)
            
            if not force:
                if response_path.exists() or sample.file_path in processed_set:
                    continue

            user_prompt = Prompt(sample.file_path, role="user")
            responses = model.generate(Prompt.combine(system_prompt, user_prompt))
            responses = response_parser.parse_llm_response(responses)
            
            (output_path / sample.label).mkdir(parents=True, exist_ok=True)
            
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)
            
            processed_list.append({'file_path': sample.file_path})
            save_checkpoint(checkpoint_file, processed_list)
            processed_count += 1
            
        except Exception as e:
            print(f"[Worker-{chunk_id}] Error: {sample.file_path}: {e}")
    
    return processed_count

def main(parser_args):
    output_path = Path("./output")
    (output_path / "malicious").mkdir(parents=True, exist_ok=True)
    (output_path / "benign").mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("Loading all samples...")
    all_samples = Dataloader(source_dir=parser_args.source_dir).load_data()
    
    samples_to_process = []
    for sample in all_samples:
        f_name = f"{Path(sample.file_path).name.removesuffix('.txt')}.json"
        if parser_args.force or not (output_path / sample.label / f_name).exists():
            samples_to_process.append(sample)
            
    total_samples = len(samples_to_process)
    print(f"Total samples to process: {total_samples}")

    if total_samples == 0:
        print("All samples processed.")
        return

    MAX_WORKERS = parser_args.max_workers
    
    chunk_size = math.ceil(total_samples / MAX_WORKERS)
    chunks = [samples_to_process[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    print(f"Starting {len(chunks)} workers on {MAX_WORKERS} processes...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk, parser_args, i, parser_args.force) for i, chunk in enumerate(chunks)]
        
        for i, future in enumerate(futures):
            try:
                count = future.result()
                print(f"Worker-{i} processed {count} samples")
            except Exception as e:
                print(f"Worker-{i} failed: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Model with Parallel Processing.")
    parser.add_argument("--config-file", type=str, default="config/deepseek-coder-6.7b.json", help="File path dẫn đến config file")
    parser.add_argument("--source-dir", type=str, default="./data/samples/")
    parser.add_argument("--system-prompt", type=str, default="src/prompts/deepseek-system-prompt.txt", help="File path đến system prompt")
    parser.add_argument("--max-workers", type=int, default=5, help="Số lượng process worker song song tối đa")
    parser.add_argument("--force", action="store_true", help="Tạo lại tất cả các mẫu ngay cả khi đầu ra đã tồn tại")
    
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main(parser.parse_args())