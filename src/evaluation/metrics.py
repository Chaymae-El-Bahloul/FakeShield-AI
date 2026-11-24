import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


@torch.no_grad()
def evaluate(model, loader, device, ablate=None, return_outputs=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    start, n = time.time(), 0
    for b in loader:
        kw = dict(input_ids=b["input_ids"].to(device),
                  attention_mask=b["attention_mask"].to(device),
                  image=b["image"].to(device))
        if ablate:
            kw["ablate"] = ablate
        probs = torch.softmax(model(**kw), dim=-1).cpu()
        y_prob.append(probs.numpy())
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(b["label"].tolist())
        n += len(b["label"])
    elapsed = time.time() - start
    y_prob = np.concatenate(y_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob) | {
        "inference_ms_per_sample": 1000 * elapsed / max(n, 1)
    }
    if return_outputs:
        return metrics, np.array(y_true), y_prob
    return metrics


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and y_prob.shape[1] == 2 and len(set(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    return m
