import json
from pathlib import Path
from utils.utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def load_predictions(base_dir="output", threshold=0.8):
    base = Path(base_dir)

    results_list = []

    for label in ["benign","malicious"]:
        for package in get_packages(base / label):
            package_path = base / label / Path(package)
            results = []
            for output_file_path in package_path.glob("*.json"):
                with open(output_file_path, mode="r") as f:
                    try: 
                        results.append(json.load(f))
                    except: 
                        continue
            final_result = calculate_result(results, threshold)
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
def calculate_result(results, threshold):
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
    is_malware = 1 if malware >=threshold else 0
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


def evaluate(results_list):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    interaction_mal_obs = 0
    for item in results_list:
        label = item['label']
        interaction_mal_obs += item.get(interaction_mal_obs,0)
        if label == 'malicious':
            if(item['is_malware'] == 1): tp+=1
            else: fp+=1
        else:
            if(item['is_malware'] == 1): fn+=1
            else: tn+=1
    acc = (tn + tp) / ( tp + tn + fp + fn)
    precision = tp / (tp + fp + 0.01)
    recall = tp / (tp + fn + 0.01)
    f1 = 2 * precision * recall / (precision + recall + 0.01)
    return {
        "accuracy": acc,
        "precision": precision,
        "f1_score": f1,
        "false_positive": fp,
        "obfuscated_rate": interaction_mal_obs / (tp + 0.01), 
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def print_report(metrics):
    print("==== Model Evaluation Report ====")
    print(f"Accuracy                   : {metrics['accuracy']:.4f}")
    print(f"Precision                   : {metrics['precision']:.4f}")
    print(f"F1 Score                   : {metrics['f1_score']:.4f}")
    print(f"False Positives            : {metrics['false_positive']}")
    print(f"Obfuscated Rate            : {metrics['obfuscated_rate']}")
    print("---- Confusion Matrix ----")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}  TN: {metrics['tn']}")
    print("===============================")

def plot_roc_from_confusion():
    plt.figure()
    for output_path in ["../output_slice", "../output_no_slice"]:
        tpr = []
        fpr = []
        labels = []
        for threshold in [0.0,0.5,0.8,1.0]:
            results = load_predictions(output_path,threshold)
            metrics = evaluate(results)
            tpr.append(metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1.e-9))
            fpr.append(metrics['fp'] / (metrics['fp'] + metrics['tn'] + 1.e-9))
            labels.append(threshold)

        # Draw ROC curve
        plt.plot(fpr, tpr, marker='o', label=output_path)
            # Label each (FPR, TPR) point
        for i in range(len(labels)):
            plt.text(fpr[i] + 0.01, tpr[i] + 0.01, str(labels[i]))

    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], '--', label="Random Guess")



    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()

    plt.savefig("roc.png")
    plt.close()



if __name__ == "__main__":
    samples = load_predictions("output")
    metrics = evaluate(samples)
    print_report(metrics)
