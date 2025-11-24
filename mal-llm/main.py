
from pre_process.data_loader import Dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed
from model.model_config import ModelConfig
from model.model import AIModel
from pathlib import Path
from eval.eval_model import *

NUM_WORKERS = 4
def set_up_model(config_file : Path):
    config = ModelConfig.from_json_file(config_file)
    model = AIModel(config)
    print(f"[INFO] Load a model: {config.model_name}")
    return model
def inference(sample, model):
    output_root = Path("../output")
    benign_out = output_root / "benign"
    malicious_out = output_root / "malicious"

    # Create output directories if they don’t exist
    benign_out.mkdir(parents=True, exist_ok=True)
    malicious_out.mkdir(parents=True, exist_ok=True)

    # Run model inference
    inputs = model.tokenize(sample.package_path)
    result = model.generate(inputs)

    # Pick correct output folder
    out_dir = benign_out if sample.label == "benign" else malicious_out
    
    # Output file path (match original sample name)
    out_file = out_dir / f"{sample.package_name}.json"

    # Write inference result
    out_file.write_text(result, encoding="utf-8")
    print(f"[INFO] Processed a package: {sample.package_name}")
if __name__ == '__main__':
    samples = Dataloader().load_data()
    model = set_up_model(Path('../config/qwen-coder-0.5b.json'))
    # for sample in samples:
    #     inference(sample,model)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(inference, s) for s in samples]
    samples = load_predictions("../output")
    metrics = evaluate(samples)
    print_report(metrics)
