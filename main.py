import argparse
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from config.model_config import ModelConfig
from src.models.model import AIModel
from src.prompts.prompt import Prompt
from data.loader import Dataloader, DataSamples
from src.utils import response_parser
from typing import List, Set

def load_checkpoints() -> Set[str]:
    output_path = Path("./output/checkpoints")

    if not output_path.exists():
        (output_path).mkdir(parents=True, exist_ok=True)
        return set()
    
    file = output_path / "checkpoint.jsonl"
    checkpoint_set = set()

    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    pkg = data.get("package") 
                    
                    if pkg:
                        checkpoint_set.add(pkg)
                        
                except json.JSONDecodeError:
                    continue

    return checkpoint_set

def save_checkpoints(package_name: str):
    output_path = Path("./output/checkpoints")
    if not output_path.exists():
        (output_path).mkdir(parents=True, exist_ok=True)

    file = output_path / "checkpoint.jsonl"

    with open(file, 'a', encoding='utf-8') as f:
        record = {"package": package_name, "status": "done"}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

def process_chunk(chunk_samples: List[DataSamples],
                  chunk_id, lock, model:AIModel=None, 
                  system_prompt:Prompt=None,
                  batch_size: int = 3) -> int:
    output_path = Path("./output")
    process_count = 0

    chunk_samples = sorted(chunk_samples, key=lambda x: x.package_length or 0, reverse=True)
    total_samples = len(chunk_samples)

    created_dirs = set()

    for i in tqdm(range(0, total_samples, batch_size), desc=f"Worker-{chunk_id}", position=chunk_id):
        current_batch = chunk_samples[i:i + batch_size]
        batch_prompts = []
        valid_samples_in_batch = []

        for sample in current_batch:
            try:
                user_prompt = Prompt(sample.package_path, role="user")
                full_prompt = Prompt.combine(system_prompt, user_prompt)
                batch_prompts.append(full_prompt)
                valid_samples_in_batch.append(sample)
            except Exception as e:
                print(f"[Worker-{chunk_id}] Error preparing prompt for {sample.package_path}: {e}")
            
        if not batch_prompts:
            continue

        try:
            successull_packages_in_batch = []
            batch_responses = model.generate_batch(batch_prompts)
            for sample, response in zip(valid_samples_in_batch, batch_responses):
                try:
                    package_path = Path(sample.package_path)
                    package_name = package_path.name.removesuffix('.txt')
                    target_dir = output_path / sample.label

                    if target_dir not in created_dirs:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(target_dir)
                    
                    response_path = target_dir / (f"{package_name}.json")
                    parsed_responses = response_parser.parse_llm_response(response)

                    with open(response_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_responses, f, indent=4, ensure_ascii=False)

                    successull_packages_in_batch.append(sample.package_name)
                    process_count += 1
                except Exception as e:
                    print(f"[Worker-{chunk_id}] Error processing response for {sample.package_path}: {e}")

            if successull_packages_in_batch:
                with lock:
                    for pkg_name in successull_packages_in_batch:
                        save_checkpoints(pkg_name)
        except Exception as e:
            print(f"[Worker-{chunk_id}] Error generating batch responses: {e}")
            continue
    
    return process_count

def main(parser_args):
    output_path = Path("./output")
    (output_path / "malicious").mkdir(parents=True, exist_ok=True)
    (output_path / "benign").mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    samples_to_process: List[DataSamples] = Dataloader(source_dir=parser_args.source_dir).load_data()

    checkpoint_set = load_checkpoints()

    samples_to_process = [
        s for s in samples_to_process if s.package_name not in checkpoint_set
    ]

    from itertools import chain
    for file in chain((output_path / "malicious").glob("*.json"), (output_path / "benign").glob("*.json")):
        package_name = file.name.removesuffix('.json')
        if package_name in checkpoint_set:
            samples_to_process = [
                s for s in samples_to_process if s.package_name != package_name
            ]
            
    total_samples = len(samples_to_process)
    if total_samples == 0:
        print("Processing complete.")
        return

    sample_sorted = sorted(samples_to_process, key=lambda x: x.package_length or 0, reverse=True)
    num_workers = min(parser_args.max_workers, total_samples)
    
    chunks = [sample_sorted[i::num_workers] for i in range(num_workers)]

    model_config = ModelConfig.from_json_file(parser_args.config_file)
    model = AIModel(model_config)
    system_prompt = Prompt(parser_args.system_prompt, role="system")

    with mp.Manager() as manager:
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk_id, chunk_samples in enumerate(chunks):
                futures.append(
                    executor.submit(
                        process_chunk,
                        chunk_samples,
                        chunk_id,
                        lock,
                        model,
                        system_prompt,
                        parser_args.batch_size
                    )
                )

            total_processed = 0
            for future in tqdm(futures, desc="Overall Progress"):
                total_processed += future.result()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalLLM Code Analysis Tool")
    parser.add_argument("--config-file", type=str, default="config/deepseek-coder-6.7b.json", help="File path dẫn đến config file")
    parser.add_argument("--source-dir", type=str, default="./data/samples/")
    parser.add_argument("--system-prompt", type=str, default="src/prompts/deepseek-system-prompt.txt", help="File path đến system prompt")
    parser.add_argument("--max-workers", type=int, default=5, help="Số lượng process worker song song tối đa")
    parser.add_argument("--batch-size", type=int, default=3, help="Số lượng mẫu xử lý trong mỗi batch")
    
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main(parser.parse_args())