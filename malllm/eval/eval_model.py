import json
from pathlib import Path
from utils.utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
def load_predictions(base_dir="output"):
    base = Path(base_dir)

    results_list = []

    for label in ["benign", "malicious"]:
        for package in get_packages(base / label):
            package_path = base / label / Path(package)
            for output_file_path in package_path.glob("*.json"):
                results = []
                with open(output_file_path, mode="r") as f:
                    results.append(json.load(f))
            final_result = calculate_result(results)
            final_result['package_name'] = package
            final_result['label'] = label
            results_list.append(final_result)
            # print(final_result)
    print(f"[INFO] Load total samples: {len(results_list)}")
    return results_list


# ---------------------------------------------------
# Classification rule:
# malicious if malware > 0.8 else benign
# (obfuscated NOT used for prediction)
# ---------------------------------------------------
def calculate_result(results):
    confidence = 0
    obfuscated = 0
    malware = 0
    securityRisk = 0
    for result in results:
        confidence = max(result.get("confidence", 0.0), confidence)
        obfuscated = max(result.get("obfuscated", 0.0),obfuscated)
        malware = max(result.get("malware", 0.0),malware)
        securityRisk = max(result.get("securityRisk", 0.0),securityRisk)
        
        # Numeric derived features
    is_malware = 1 if malware >= 0.8 else 0
    is_obfuscated = 1 if obfuscated > 0.5 else 0
    threat_score = 0.6 * malware + 0.3 * obfuscated + 0.1 * securityRisk
    interaction_mal_obs = 1 if malware >= 0.8 and obfuscated >= 0.5 else 0
    return {
        "confidence" : confidence,
        "obfucated_score" : obfuscated,
        "malware_score": malware,
        "securityRisk" : securityRisk,
        "is_obfucated" : is_obfuscated,
        "is_malware": is_malware,
        "threat_score" : threat_score,
        "interaction_mal_obs" : interaction_mal_obs
    }


def evaluate(results_list, threshold=0.5):
    y_true = []
    y_pred = []
    interaction_mal_obs = 0
    for item in results_list:
        label = item['label']
        score = item.get('malware_score', 0.0)
        interaction_mal_obs += item.get(interaction_mal_obs,0)
        # Convert label to binary
        y_true.append(1 if label.lower() == 'malicious' else 0)
        # Prediction based on threshold
        y_pred.append(1 if score >= threshold else 0)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "f1_score": f1,
        "false_positive": false_positive_rate,
        "interaction_mal_obs" : round(interaction_mal_obs / len(list(filter(lambda x : x == 1,y_true))),2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def print_report(metrics):
    print("==== Model Evaluation Report ====")
    print(f"Accuracy                   : {metrics['accuracy']:.4f}")
    print(f"F1 Score                   : {metrics['f1_score']:.4f}")
    print(f"False Positives            : {metrics['false_positive']}")
    print(f"Obfuscated_rate            : {metrics['interaction_mal_obs']}")
    print("---- Confusion Matrix ----")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}  TN: {metrics['tn']}")
    print("===============================")


if __name__ == "__main__":
    samples = load_predictions("output")
    metrics = evaluate(samples)
    print_report(metrics)
