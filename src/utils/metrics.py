import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

def get_metric(preds, labels, task, metric):
    # Force predictions to be a NumPy array to prevent indexing errors
    preds = np.array(preds)
    labels = np.array(labels)
    
    if task == 'multi-class':
        if metric == 'Accuracy':
            return accuracy_score(labels, np.argmax(preds, axis=1))
        elif metric == 'Bal. Acc.':
            return balanced_accuracy_score(labels, np.argmax(preds, axis=1))
        elif metric == 'AUROC':
            # Multi-class requires the entire probability matrix
            all_classes = np.arange(preds.shape[1])
            return roc_auc_score(labels, preds, multi_class="ovr", labels=all_classes)
        elif metric == 'F1 Score':
            return f1_score(labels, np.argmax(preds, axis=1), average="macro")
            
    elif task == 'binary':
        if metric == 'Accuracy':
            return accuracy_score(labels, np.argmax(preds, axis=1))
        elif metric == 'AUROC':
            # Binary requires only the probability of the positive class (index 1)
            return roc_auc_score(labels, preds[:, 1])
        elif metric == 'F1 Score':
            return f1_score(labels, np.argmax(preds, axis=1))
        elif metric == 'Matt. Corr.':
            return matthews_corrcoef(labels, np.argmax(preds, axis=1))
            
    raise ValueError(f"Unknown combination of task {task} and metric {metric}")


# Modified by Tang Yan Vei
# Date: 2026-05-05

# Changes:
# - Force the predictions to be a NumPy array to prevent indexing errors 

# Original project:
# AutoPrognosis-M
# Licensed under the Apache License 2.0.