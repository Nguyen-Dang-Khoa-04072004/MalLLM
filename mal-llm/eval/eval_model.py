import json
from pathlib import Path

def load_predictions(base_dir="output"):
    base = Path(base_dir)

    samples = []

    for label in ["benign", "malicious"]:
        folder = base / label
        if not folder.exists():
            print(f"[ERROR] Folder is not exists: {folder}")
            continue

        for file in folder.glob("*.json"):
            try:
                data = json.loads(file.read_text())
            except Exception as e:
                print(f"[ERROR] Error reading {file}: {e}")
                continue

            samples.append({
                "file": file.name,
                "true_label": label,
                "malware": data.get("malware", 0.0),
                "obfuscated": data.get("obfuscated", 0.0),
            })
    print(f"[INFO] Load total samples: {len(samples)}")
    return samples


# ---------------------------------------------------
# Classification rule:
# malicious if malware > 0.8 else benign
# (obfuscated NOT used for prediction)
# ---------------------------------------------------
def predict_sample(malware_score):
    return "malicious" if malware_score > 0.8 else "benign"


def evaluate(samples):
    true_labels = []
    pred_labels = []

    # Counters
    tp = 0   # true malicious predicted malicious
    fp = 0   # benign predicted malicious
    tn = 0   # benign predicted benign
    fn = 0   # malicious predicted benign

    obfuscated_count = 0

    for s in samples:
        true_label = s["true_label"]
        pred_label = predict_sample(s["malware"])

        # Update confusion matrix
        if true_label == "malicious":
            if pred_label == "malicious":
                tp += 1
            else:
                fn += 1
        else:  # true_label == "benign"
            if pred_label == "malicious":
                fp += 1
            else:
                # Count obfuscated samples in malicious code
                if s["obfuscated"] > 0.8:
                    obfuscated_count += 1
                tn += 1

    # --------------------------
    # Manual metric computation
    # --------------------------
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0)

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "false_positive": fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "obfuscated_samples": obfuscated_count,
        "total_samples": total
    }


def print_report(metrics):
    print("==== Model Evaluation Report ====")
    print(f"Total Samples              : {metrics['total_samples']}")
    print(f"Accuracy                   : {metrics['accuracy']:.4f}")
    print(f"F1 Score                   : {metrics['f1_score']:.4f}")
    print(f"False Positives            : {metrics['false_positive']}")
    print(f"Rate bbfuscated code / malicious code   : {(metrics['obfuscated_samples'] /( metrics['tn'] + 1)):.2f}")
    print("---- Confusion Matrix ----")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}  TN: {metrics['tn']}")
    print("===============================")


if __name__ == "__main__":
    samples = load_predictions("output")
    metrics = evaluate(samples)
    print_report(metrics)
