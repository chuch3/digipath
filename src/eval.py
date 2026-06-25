from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


# Mainly used for plot labels
def _idx_to_class_name(label_map):
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
    Runs the model with the loader and collects:
    - y_true: ground-truth integer labels
    - y_score: predicted softmax probabilities
    - y_pred: predicted integer labels
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


# ROC-AUC curves (multi-class, so one-vs-rest)
def plot_roc_auc(model, loader, label_map, device, save_path, title="ROC-AUC Curve"):
    class_names = _idx_to_class_name(label_map)
    num_classes = len(class_names)

    y_true, y_score, _ = _collect_predictions(model, loader, device)

    # Binarize true labels: shape (N, num_classes)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    # label_binarize collapses to a single column when num_classes == 2
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
    plt.show()

    print(f"=> ROC-AUC curve saved to {save_path}")
    return roc_auc_val


# Groups the distinct classes by superclass ("aca", "scc", "nor")
def _group_to_parent_class(class_names, y_true, y_pred):
    prefix_of = [name.split("_")[0] for name in class_names]

    # Stable, deduplicated parent order: first-seen order of appearance
    seen = []
    for p in prefix_of:
        if p not in seen:
            seen.append(p)
    parent_names = seen

    parent_idx = {name: i for i, name in enumerate(parent_names)}
    old_to_new = np.array(
        [parent_idx[prefix_of[old_i]] for old_i in range(len(class_names))]
    )

    y_true_grouped = old_to_new[y_true]
    y_pred_grouped = old_to_new[y_pred]

    return parent_names, y_true_grouped, y_pred_grouped


# Confusion matrix
def plot_confusion_matrix(
    model,
    loader,
    label_map,
    device,
    save_path,
    title="Confusion Matrix",
    group_subtypes=True,
):
    class_names = _idx_to_class_name(label_map)
    num_classes = len(class_names)

    y_true, _, y_pred = _collect_predictions(model, loader, device)

    if group_subtypes:
        class_names, y_true, y_pred = _group_to_parent_class(
            class_names, y_true, y_pred
        )
        num_classes = len(class_names)

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
    plt.show()

    print(f"=> Confusion matrix saved to {save_path}")
    return cm


# Loss / accuracy evolution
def plot_training_curves(
    train_loss_history,
    train_acc_history,
    best_epoch,
    save_path,
    val_loss_history=None,
    val_acc_history=None,
    title="Training Curves",
):
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
    axes[0].axvline(x=best_epoch, ls="--", label="best epoch", color="b")
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
    axes[1].axvline(x=best_epoch, ls="--", label="best epoch", color="b")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"=> Training curves saved to {save_path}")


# Wrapper to run all three evaluations at once
def run_full_evaluation(
    model,
    test_loader,
    label_map,
    device,
    best_epoch,
    loss_epochs,
    acc_epochs,
    output_dir,
    val_loss_epochs=None,
    val_acc_epochs=None,
    label="",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_auc(
        model,
        test_loader,
        label_map,
        device,
        save_path=output_dir / f"roc_auc_curve_{label}.png",
        title="ROC-AUC Curve (Test Set)",
    )

    plot_confusion_matrix(
        model,
        test_loader,
        label_map,
        device,
        save_path=output_dir / f"confusion_matrix_{label}.png",
        title="Confusion Matrix (Test Set)",
    )

    plot_training_curves(
        loss_epochs,
        acc_epochs,
        best_epoch,
        save_path=output_dir / f"loss_acc_curves_{label}.png",
        val_loss_history=val_loss_epochs,
        val_acc_history=val_acc_epochs,
        title="Training & Validation Curves",
    )
