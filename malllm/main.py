from pre_process.data_loader import Dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from model.model_config import ModelConfig
from model.model import AIModel
from pathlib import Path
from eval.eval_model import *
from utils.utils import *
import random
import threading
import json

NUM_WORKERS = 4

# =============================
#   CHECKPOINT IMPLEMENTATION
# =============================

CHECKPOINT_FILE = Path("../output/__checkpoint__/done.txt")
CHECKPOINT_LOCK = threading.Lock()

def load_checkpoint():
    """Load list of processed items."""
    if not CHECKPOINT_FILE.exists():
        return set()

    with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())


def save_checkpoint(key: str):
    """Append a processed item to checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CHECKPOINT_LOCK:
        with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
            f.write(key + "\n")


def is_processed(key: str, checkpoint: set):
    return key in checkpoint


# =============================
#    MODEL SETUP
# =============================

def set_up_model(config_file: Path):
    config = ModelConfig.from_json_file(config_file)
    model = AIModel(config)
    print(f"[INFO] Loaded model: {config.model_name}")
    return model


# =============================
#       INFERENCE
# =============================

def inference(sample, model, checkpoint):
    output_root = Path("../output")

    # Unique key for checkpoint: "label/package/file"
    key = f"{sample.label}/{sample.package_name}/{sample.name}"

    if is_processed(key, checkpoint):
        print(f"[SKIP] Already processed: {key}")
        return

    out_pkg_dir = output_root / sample.label / sample.package_name
    out_pkg_dir.mkdir(parents=True, exist_ok=True)

    try:
        inputs = model.tokenize(Path(sample.package_path))
        result = model.generate(inputs)

        if result is not None and result.strip() != "":
            json_output_path = out_pkg_dir / f"{sample.name[:-3]}json"
            json_output_path.write_text(extract_json_string(result), encoding="utf-8")

        print(f"[INFO] Processed slice: {sample.package_name}/{sample.name}")

        # Save checkpoint
        save_checkpoint(key)

    except Exception as e:
        print(f"[ERROR] Failed to process {sample.name} in {sample.package_name}: {e}")


# =============================
#        MAIN EXECUTION
# =============================

if __name__ == '__main__':
    # Load samples using your existing Dataloader
    samples = Dataloader().load_data()

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"[INFO] Loaded checkpoint: {len(checkpoint)} items already processed")

    # Load model
    model = set_up_model(Path('../config/deepseek-coder-6.7b.json'))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(inference, s, model, checkpoint) 
            for s in samples
        ]

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("[THREAD ERROR]", e)

    # Evaluation
    samples = load_predictions("../output")
    metrics = evaluate(samples)
    print_report(metrics)
