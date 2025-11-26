from eval.eval_model import *
samples = load_predictions("../output")
metrics = evaluate(samples)
print_report(metrics)