import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


from typing import Dict, Optional, Sequence


@torch.no_grad()
def multiclass_metrics(
    output: torch.Tensor,                 # [N, C], raw logits
    labels: torch.Tensor,                 # [N], integer class indices
    *,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,

    # Turn individual metrics on/off
    accuracy: bool = True,
    balanced_accuracy: bool = True,
    macro_f1: bool = True,
    weighted_f1: bool = True,
    classwise_f1: bool = True,
    classwise_precision: bool = True,
    classwise_recall: bool = True,
    confusion_matrix: bool = True,
    topk: Optional[Sequence[int]] = None,
    mcc: bool = False,

    zero_division: float = 0.0,
) -> Dict:
    """
    Computes single-label multiclass classification metrics.

    Notes
    -----
    - `output` should contain logits with shape [N, C].
    - `labels` should contain class indices with shape [N].
    - For reliable F1 scores, pass predictions from the complete epoch.
    """
    if output.ndim != 2:
        raise ValueError(f"output must have shape [N, C], got {output.shape}")

    labels = labels.reshape(-1).to(device=output.device, dtype=torch.long)

    if output.shape[0] != labels.numel():
        raise ValueError(
            f"Sample mismatch: output has {output.shape[0]} samples, "
            f"labels has {labels.numel()}."
        )

    C = output.shape[1] if num_classes is None else num_classes

    if C != output.shape[1]:
        raise ValueError(
            f"num_classes={C}, but output has {output.shape[1]} classes."
        )

    if class_names is not None and len(class_names) != C:
        raise ValueError(
            f"Expected {C} class names, received {len(class_names)}."
        )

    if labels.numel() == 0:
        raise ValueError("Cannot compute metrics from empty tensors.")

    if labels.min() < 0 or labels.max() >= C:
        raise ValueError(f"Labels must be in the range [0, {C - 1}].")

    predictions = output.argmax(dim=1)

    # Confusion matrix: rows = targets, columns = predictions
    indices = labels * C + predictions
    cm = torch.bincount(indices, minlength=C * C).reshape(C, C)

    cm_float = cm.to(torch.float64)
    tp = cm_float.diag()
    support = cm_float.sum(dim=1)          # True samples per class
    predicted_count = cm_float.sum(dim=0)  # Predicted samples per class
    total = cm_float.sum()

    def safe_divide(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        fill = torch.full_like(numerator, float(zero_division))
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            fill,
        )

    precision = safe_divide(tp, predicted_count)
    recall = safe_divide(tp, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    results = {}

    if accuracy:
        results["accuracy"] = (tp.sum() / total).item()

    if balanced_accuracy:
        # Mean recall over classes present in the targets.
        valid = support > 0
        results["balanced_accuracy"] = recall[valid].mean().item()

    if macro_f1:
        # Mean F1 over classes present in targets or predictions.
        valid = (support + predicted_count) > 0
        results["macro_f1"] = f1[valid].mean().item()

    if weighted_f1:
        results["weighted_f1"] = (
            (f1 * support).sum() / support.sum().clamp_min(1)
        ).item()

    if classwise_f1:
        results["classwise_f1"] = format_classwise(
            f1, class_names
        )

    if classwise_precision:
        results["classwise_precision"] = format_classwise(
            precision, class_names
        )

    if classwise_recall:
        results["classwise_recall"] = format_classwise(
            recall, class_names
        )

    if confusion_matrix:
        results["confusion_matrix"] = cm.cpu()

    if topk is not None:
        for k in topk:
            if not 1 <= k <= C:
                raise ValueError(f"top-{k} is invalid for {C} classes.")

            top_predictions = output.topk(k, dim=1).indices
            correct = top_predictions.eq(labels[:, None]).any(dim=1)
            results[f"top_{k}_accuracy"] = correct.float().mean().item()

    if mcc:
        results["mcc"] = multiclass_mcc(cm_float).item()

    return results


def format_classwise(
    values: torch.Tensor,
    class_names: Optional[Sequence[str]] = None,
):
    """Returns classwise values as a tensor or a name-to-value dictionary."""
    values = values.detach().cpu()

    if class_names is None:
        return values

    return {
        name: values[index].item()
        for index, name in enumerate(class_names)
    }


def multiclass_mcc(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """Multiclass Matthews correlation coefficient."""
    cm = confusion_matrix.to(torch.float64)

    correct = cm.diag().sum()
    total = cm.sum()
    true_count = cm.sum(dim=1)
    predicted_count = cm.sum(dim=0)

    numerator = correct * total - torch.dot(true_count, predicted_count)

    denominator = torch.sqrt(
        (total.square() - predicted_count.square().sum())
        * (total.square() - true_count.square().sum())
    )

    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )



@torch.no_grad()
def correct_wrong_indices(
    logits: torch.Tensor,   # [B, C]
    labels: torch.Tensor,   # [B]
):
    """
    Returns batch indices and predicted class indices for correct
    and incorrect top-1 predictions.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, C], got {logits.shape}")

    labels = labels.reshape(-1).to(logits.device, dtype=torch.long)

    if logits.shape[0] != labels.numel():
        raise ValueError("Batch sizes of logits and labels do not match.")

    predictions = logits.argmax(dim=1)
    correct_mask = predictions.eq(labels)
    wrong_mask = ~correct_mask

    return {
        # Positions within the batch
        "correct_sample_indices": correct_mask.nonzero(as_tuple=True)[0],
        "wrong_sample_indices": wrong_mask.nonzero(as_tuple=True)[0],

        # Predicted classes at those positions
        "correct_class_indices": predictions[correct_mask],
        "wrong_class_indices": predictions[wrong_mask],

        # Ground-truth classes for the incorrect samples
        "wrong_true_class_indices": labels[wrong_mask],
    }


def roc_separation(x1, x2, plot=True):
    """
    x1, x2: 1-D torch tensors.

    Tests both:
        x1 > x2  -> score = value
        x1 < x2  -> score = -value

    Returns the better orientation and the threshold
    maximizing Youden's J = TPR - FPR.
    """
    x1 = x1.detach().float().cpu().numpy().ravel()
    x2 = x2.detach().float().cpu().numpy().ravel()

    # x1 = positive class, x2 = negative class
    y = np.concatenate([
        np.ones(len(x1)),
        np.zeros(len(x2))
    ])

    values = np.concatenate([x1, x2])

    # -----------------------------------
    # Direction 1: larger -> x1
    # -----------------------------------
    auc_high = roc_auc_score(y, values)

    # Direction 2: smaller -> x1
    auc_low = roc_auc_score(y, -values)

    # Choose better direction
    if auc_high >= auc_low:
        scores = values
        direction = "x1 >= threshold"
        auc = auc_high
        sign = 1
    else:
        scores = -values
        direction = "x1 <= threshold"
        auc = auc_low
        sign = -1

    # ROC
    fpr, tpr, thresholds = roc_curve(y, scores)

    # Best threshold: Youden's J
    J = tpr - fpr
    idx = np.argmax(J)

    score_threshold = thresholds[idx]

    # Convert threshold back to original value scale
    threshold = score_threshold if sign == 1 else -score_threshold

    print(f"Direction       : {direction}")
    print(f"AUROC           : {auc:.4f}")
    print(f"Best threshold  : {threshold:.6f}")
    print(f"TPR             : {tpr[idx]:.4f}")
    print(f"FPR             : {fpr[idx]:.4f}")
    print(f"Youden's J      : {J[idx]:.4f}")

    if plot:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.scatter(fpr[idx], tpr[idx],
                    label=f"threshold = {threshold:.3f}")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return {
        "auc": auc,
        "threshold": threshold,
        "direction": direction,
        "tpr": tpr[idx],
        "fpr": fpr[idx],
        "youden_j": J[idx],
    }