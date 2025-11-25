from pre_process.data_loader import Dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from model.model_config import ModelConfig
from model.model import AIModel
from pathlib import Path
from eval.eval_model import *
import random
NUM_WORKERS = 8

def set_up_model(config_file: Path):
    config = ModelConfig.from_json_file(config_file)
    model = AIModel(config)
    print(f"[INFO] Loaded model: {config.model_name}")
    return model

def inference(sample, model):
    output_root = Path("../output")
    benign_out = output_root / "benign"
    malicious_out = output_root / "malicious"

    benign_out.mkdir(parents=True, exist_ok=True)
    malicious_out.mkdir(parents=True, exist_ok=True)

    try:
        inputs = model.tokenize(sample.package_path)
        result = model.generate(inputs)

        out_dir = benign_out if sample.label == "benign" else malicious_out
        out_file = out_dir / f"{sample.package_name}.json"

        out_file.write_text(result, encoding="utf-8")
        print(f"[INFO] Processed package: {sample.package_name}")

    except Exception as e:
        print(f"[ERROR] Failed to process {sample.package_name}: {e}")

if __name__ == '__main__':
    samples = Dataloader().load_data()
    model = set_up_model(Path('../config/deepseek-coder-6.7b.json'))
    random.shuffle(samples)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(inference, s, model) for s in samples]

        # Optional: show errors from workers
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("[THREAD ERROR]", e)

    samples = load_predictions("../output")
    metrics = evaluate(samples)
    print_report(metrics)
