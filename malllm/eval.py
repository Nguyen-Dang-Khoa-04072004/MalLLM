from eval.eval_model import *
from  utils.utils import *
import pandas as pd

results_list = load_predictions("../output_no_slice")
metrics = evaluate(results_list)
print_report(metrics)
# plot_roc_from_confusion()
