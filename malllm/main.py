from pre_process.data_loader import Dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from model.model_config import ModelConfig
from model.model import AIModel
from pathlib import Path
from eval.eval_model import *
from utils.utils import *
import random

NUM_WORKERS = 1


def set_up_model(config_file: Path):
    config = ModelConfig.from_json_file(config_file)
    model = AIModel(config)
    print(f"[INFO] Loaded model: {config.model_name}")
    return model


def inference(sample, model):
    output_root = Path("../output")

    # Example: ../output/benign/pkgA
    out_pkg_dir = output_root / sample.label / sample.package_name
    out_pkg_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Tokenize → generate prediction
        inputs = model.tokenize(Path(sample.package_path))
        result = model.generate(inputs)

        # Output one JSON per text slice
        if(result != None or result.replace(" ","") != ""):
            json_output_path = out_pkg_dir / f"{sample.name[:-3]}json"
            json_output_path.write_text(extract_json_string(result), encoding="utf-8")

        print(f"[INFO] Processed slice: {sample.package_name}/{sample.name}")

    except Exception as e:
        print(f"[ERROR] Failed to process {sample.name} in {sample.package_name}: {e}")


if __name__ == '__main__':
    # Load samples using your existing Dataloader
    samples = Dataloader().load_data()
    # Load model
    model = set_up_model(Path('../config/deepseek-coder-6.7b.json'))

    # Multithreaded inference
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(inference, s, model) for s in samples]

        # Optional: catch worker-level errors
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("[THREAD ERROR]", e)

    # Evaluation (your existing code)
    samples = load_predictions("../output")
    metrics = evaluate(samples)
    print_report(metrics)
