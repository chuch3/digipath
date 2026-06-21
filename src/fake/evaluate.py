"""
evaluate.py

Evaluation/plotting utilities for the lung classification pipeline in train.py.

Produces three PNGs:
    1. ROC-AUC curves (one-vs-rest, per class + micro/macro average)   -> roc_auc_curve.png
    2. Confusion matrix heatmap (counts + row-normalized %)            -> confusion_matrix.png
    3. Loss / accuracy evolution over epochs (train vs. val)           -> loss_acc_curves.png

Designed to be called directly from train.py's train() function, reusing
objects that already exist there: model, test_loader, valid_loader,
label_map, loss_epochs, acc_epochs, device.

No assumptions are made about ClassifierHead's internals beyond it being an
nn.Module that returns logits of shape (batch, num_classes) given an
embedding batch of shape (batch, embed_dim) -- exactly how it's already
used in train.py's training/validation loops.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _idx_to_class_name(label_map):
    """
    label_map, as built upstream in dataset.py/load_dataset(), is assumed to
    map {class_name: class_idx}. We invert it here to get plot-friendly
    class names in the correct index order. If label_map already maps
    {idx: name}, this still works since we sort by the resulting keys.
    """
    # Detect orientation: values are ints -> {name: idx}; else {idx: name}
    sample_val = next(iter(label_map.values()))
    if isinstance(sample_val, (int, np.integer)):
        idx_to_name = {v: k for k, v in label_map.items()}
    else:
        idx_to_name = dict(label_map)

    num_classes = len(idx_to_name)
    return [str(idx_to_name[i]) for i in range(num_classes)]


@torch.inference_mode()
def _collect_predictions(model, loader, device):
    """
    Runs the model over a loader and collects:
        y_true  : (N,)            ground-truth integer labels
        y_score : (N, num_classes) softmax probabilities
        y_pred  : (N,)            argmax predicted labels
    """
    model.eval()

    all_true, all_scores = [], []

    for feat_batch, label_batch in loader:
        feat_batch = feat_batch.to(device)
        logits = model(feat_batch)
        probs = torch.softmax(logits, dim=1)

        all_true.append(label_batch.detach().cpu())
        all_scores.append(probs.detach().cpu())

    y_true = torch.cat(all_true).numpy()
    y_score = torch.cat(all_scores).numpy()
    y_pred = y_score.argmax(axis=1)

    return y_true, y_score, y_pred


# --------------------------------------------------------------------------- #
# 1. ROC-AUC curves (multi-class, one-vs-rest)
# --------------------------------------------------------------------------- #
def plot_roc_auc(model, loader, label_map, device, save_path, title="ROC-AUC Curve"):
    """
    Multi-class ROC via one-vs-rest. Plots one curve per class plus
    micro-average and macro-average curves, each annotated with its AUC.
    """
    class_names = _idx_to_class_name(label_map)
    num_classes = len(class_names)

    y_true, y_score, _ = _collect_predictions(model, loader, device)

    # Binarize true labels: shape (N, num_classes)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    # label_binarize collapses to a single column when num_classes == 2;
    # guard for that edge case even though this module targets multi-class.
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc_val = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_val[i] = auc(fpr[i], tpr[i])

    # Micro-average: flatten all classes together
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_val["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average: interpolate each class curve onto a common grid, then average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc_val["macro"] = auc(fpr["macro"], tpr["macro"])

    # --- Plot ---
    plt.figure(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_classes, 1)))

    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=1.8,
            label=f"{class_names[i]} (AUC = {roc_auc_val[i]:.3f})",
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        color="deeppink",
        linestyle=":",
        lw=2.5,
        label=f"micro-average (AUC = {roc_auc_val['micro']:.3f})",
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        color="navy",
        linestyle=":",
        lw=2.5,
        label=f"macro-average (AUC = {roc_auc_val['macro']:.3f})",
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"=> ROC-AUC curve saved to {save_path}")
    return roc_auc_val


# --------------------------------------------------------------------------- #
# 2. Confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    model, loader, label_map, device, save_path, title="Confusion Matrix"
):
    """
    Plots a confusion matrix with raw counts and row-normalized percentages
    annotated together in each cell.
    """
    class_names = _idx_to_class_name(label_map)
    num_classes = len(class_names)

    y_true, _, y_pred = _collect_predictions(model, loader, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    annot = np.array(
        [
            [f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)" for j in range(num_classes)]
            for i in range(num_classes)
        ]
    )

    plt.figure(figsize=(max(6, num_classes * 1.2), max(5, num_classes * 1.0)))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"=> Confusion matrix saved to {save_path}")
    return cm


# --------------------------------------------------------------------------- #
# 3. Loss / accuracy evolution
# --------------------------------------------------------------------------- #
def plot_training_curves(
    train_loss_history,
    train_acc_history,
    save_path,
    val_loss_history=None,
    val_acc_history=None,
    title="Training Curves",
):
    """
    Plots loss and accuracy side by side over epochs.

    train.py currently only persists loss_epochs/acc_epochs for training.
    If you also want validation curves on this same plot, accumulate
    val_loss/val_acc per epoch into lists inside the training loop in
    train.py (see suggested patch below) and pass them in here.
    """
    epochs_range = range(1, len(train_loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    axes[0].plot(epochs_range, train_loss_history, label="Train loss", color="tab:blue")
    if val_loss_history is not None:
        axes[0].plot(
            epochs_range, val_loss_history, label="Val loss", color="tab:orange"
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss evolution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy subplot
    axes[1].plot(
        epochs_range,
        np.array(train_acc_history) * 100,
        label="Train acc",
        color="tab:green",
    )
    if val_acc_history is not None:
        axes[1].plot(
            epochs_range,
            np.array(val_acc_history) * 100,
            label="Val acc",
            color="tab:red",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy evolution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"=> Training curves saved to {save_path}")


# --------------------------------------------------------------------------- #
# Convenience wrapper: run all three evaluations at once
# --------------------------------------------------------------------------- #
def run_full_evaluation(
    model,
    test_loader,
    label_map,
    device,
    loss_epochs,
    acc_epochs,
    output_dir,
    val_loss_epochs=None,
    val_acc_epochs=None,
):
    """
    Call this once from train.py after the training loop finishes, e.g.:

        from evaluate import run_full_evaluation
        run_full_evaluation(
            model, test_loader, label_map, device,
            loss_epochs, acc_epochs, output_dir=LUNG_MODEL_DIR,
        )

    Returns a dict with the computed ROC-AUC scores and confusion matrix
    in case you want to log or assert on them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_auc_val = plot_roc_auc(
        model,
        test_loader,
        label_map,
        device,
        save_path=output_dir / "roc_auc_curve.png",
        title="ROC-AUC Curve (Test Set)",
    )

    cm = plot_confusion_matrix(
        model,
        test_loader,
        label_map,
        device,
        save_path=output_dir / "confusion_matrix.png",
        title="Confusion Matrix (Test Set)",
    )

    plot_training_curves(
        loss_epochs,
        acc_epochs,
        save_path=output_dir / "loss_acc_curves.png",
        val_loss_history=val_loss_epochs,
        val_acc_history=val_acc_epochs,
        title="Training / Validation Curves",
    )

    return {"roc_auc": roc_auc_val, "confusion_matrix": cm}
