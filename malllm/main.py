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
import sys
import time
import os
import glob

# =============================
#   CẤU HÌNH KAGGLE
# =============================

NUM_WORKERS = 4
# Kaggle chỉ cho phép ghi vào thư mục này
OUTPUT_ROOT = Path("/kaggle/working/output") 
CHECKPOINT_FILE = OUTPUT_ROOT / "__checkpoint__/done.txt"
CHECKPOINT_LOCK = threading.Lock()

# Thời gian bắt đầu chạy
START_TIME = time.time()
# Giới hạn an toàn: 11.5 tiếng (đổi ra giây) để kịp Save Version
TIME_LIMIT = 11.5 * 3600 

def check_time_limit():
    """Kiểm tra xem sắp hết giờ chưa"""
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        print(f"[STOP] Reached time limit ({elapsed/3600:.2f} hours). Stopping safely.")
        return True
    return False

# =============================
#   CHECKPOINT LOGIC (NÂNG CẤP)
# =============================

def load_checkpoint():
    """
    Load checkpoint từ 2 nguồn:
    1. Checkpoint của phiên đang chạy (nếu có).
    2. Checkpoint từ các phiên chạy trước (nếu bạn add output cũ làm input).
    """
    processed_files = set()
    
    # 1. Quét tất cả các file done.txt có trong Input (từ các phiên chạy trước)
    # Kaggle mount input tại /kaggle/input
    input_checkpoints = glob.glob("/kaggle/input/**/done.txt", recursive=True)
    for cp_path in input_checkpoints:
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
                processed_files.update(lines)
            print(f"[INFO] Loaded {len(lines)} items from old run: {cp_path}")
        except Exception as e:
            print(f"[WARN] Could not read input checkpoint {cp_path}: {e}")

    # 2. Load checkpoint hiện tại (nếu resume trong cùng session hoặc file đã tồn tại)
    if CHECKPOINT_FILE.exists():
        with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            processed_files.update(lines)
            
    return processed_files

def save_checkpoint(key: str):
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
#        INFERENCE
# =============================

def inference(sample, model, checkpoint):
    # Kiểm tra đường dẫn output
    output_root = OUTPUT_ROOT

    key = f"{sample.label}/{sample.package_name}/{sample.name}"

    if is_processed(key, checkpoint):
        # print(f"[SKIP] Already processed: {key}") # Comment bớt để đỡ spam log trên Kaggle
        return

    out_pkg_dir = output_root / sample.label / sample.package_name
    out_pkg_dir.mkdir(parents=True, exist_ok=True)

    try:
        inputs = model.tokenize(Path(sample.package_path))
        result = model.generate(inputs)

        if result is not None and result.strip() != "":
            json_output_path = out_pkg_dir / f"{sample.name[:-3]}json"
            json_output_path.write_text(extract_json_string(result), encoding="utf-8")

        print(f"[INFO] Processed: {sample.package_name}/{sample.name}")
        save_checkpoint(key)

    except Exception as e:
        print(f"[ERROR] Failed {sample.name}: {e}")

# =============================
#        MAIN EXECUTION
# =============================

if __name__ == '__main__':
    # 1. Setup thư mục
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Data & Checkpoint
    samples = Dataloader().load_data()
    checkpoint = load_checkpoint()
    
    # Tính toán
    total_samples = len(samples)
    remaining_samples = [s for s in samples if f"{s.label}/{s.package_name}/{s.name}" not in checkpoint]
    print(f"[INFO] Total: {total_samples}, Done: {len(checkpoint)}, Remaining: {len(remaining_samples)}")

    if not remaining_samples:
        print("All tasks completed! Exiting.")
        sys.exit(0)

    # 3. Load Model
    # Đảm bảo đường dẫn config đúng trên Kaggle (thường nằm trong /kaggle/input/...)
    model = set_up_model(Path('../config/deepseek-coder-6.7b.json'))

    # 4. Chạy Executor
    # Lưu ý: Không submit toàn bộ samples vào executor ngay để kiểm soát thời gian tốt hơn
    # Chúng ta sẽ submit từng batch nhỏ hoặc check time trong vòng lặp
    
    stop_event = threading.Event()

    def wrapped_inference(s):
        if stop_event.is_set(): return
        inference(s, model, checkpoint)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for s in remaining_samples:
            if check_time_limit():
                stop_event.set()
                break
            futures.append(executor.submit(wrapped_inference, s))
        
        # Đợi các task đang chạy dở hoàn thành
        for f in as_completed(futures):
            pass

    print("[DONE] Session finished (Completed or Time Limit Reached).")
