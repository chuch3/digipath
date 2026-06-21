import os
import unittest
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.models import ViT_B_16_Weights

from constant import (
    LUNG_IMAGES_DIR,
    LUNG_LOADED_FILE,
    LUNG_METADATA_FILE,
    LUNG_MODEL_DIR,
)
from dataset import load_dataset
from model import ClassifierHead
from transform import det_transform


class ViTWithHead(nn.Module):
    """
    Combines a frozen ViT backbone (feature extractor only, no internal head)
    with a trainable ClassifierHead.

    The backbone's internal classification head is bypassed: we extract the
    [CLS] token embedding directly from the encoder output and pass it through
    our own head. This is required so that Grad-CAM gradients flow through the
    correct path (backbone encoder → MLP head) rather than through ViT's own
    pre-trained head.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate torchvision ViT forward up to the encoder output,
        # then take the [CLS] token and pass through our head.
        # This keeps the backbone frozen but lets gradients flow back
        # through the encoder layers for Grad-CAM.
        b = self.backbone

        x = b._process_input(x)  # patch embedding + position encoding
        cls = b.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend [CLS]
        x = b.encoder(x)  # transformer blocks
        cls_out = x[:, 0]  # [CLS] token  (B, 768)
        return self.head(cls_out)  # (B, num_classes)


class GradCAM:
    """
    Grad-CAM for a frozen ViT backbone + MLP ClassifierHead.

    Hooks the last transformer encoder block. Gradients come from the
    MLP head's output, so the map reflects what matters for the *custom*
    classifier, not the original ViT head.

    Usage:
        model = ViTWithHead(backbone, head)
        cam = GradCAM(model)
        heatmap = cam(img_tensor, class_idx)   # (224, 224) float32 in [0,1]
        cam.remove_hooks()
    """

    def __init__(self, model: ViTWithHead):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

        # Hook the last encoder block of the backbone inside ViTWithHead
        target = model.backbone.encoder.layers[-1]
        self._fwd_hook = target.register_forward_hook(self._save_acts)
        self._bwd_hook = target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _module, _input, output):
        # output: (B, num_tokens, embed_dim)
        self._acts = output.detach()

    def _save_grads(self, _module, _grad_input, grad_output):
        # grad_output[0]: (B, num_tokens, embed_dim)
        self._grads = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self,
        img_tensor: torch.Tensor,  # (1, 3, 224, 224)
        class_idx: int,
    ) -> np.ndarray:  # (224, 224) heatmap in [0, 1]

        self.model.eval()
        # Unfreeze input tensor for gradient tracking; backbone weights stay frozen.
        img_tensor = img_tensor.detach().requires_grad_(True)

        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # acts / grads: (1, 197, 768)  → drop CLS token → (196, 768)
        acts = self._acts[0, 1:]  # patch tokens only
        grads = self._grads[0, 1:]

        # Global-average-pool the gradients over the embedding dim → (196,)
        weights = grads.mean(dim=-1)
        cam = (weights[:, None] * acts).sum(dim=-1)  # (196,)
        cam = F.relu(cam)

        n_patches = int(cam.shape[0] ** 0.5)  # 14 for ViT-B/16
        cam = cam.reshape(n_patches, n_patches).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (224, 224), resample=Image.BILINEAR
        )
        return np.array(cam_img) / 255.0

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class GuidedBackprop:
    """
    Gradient-based saliency for ViT (input-gradient variant).

    Note: ViT uses GELU activations and softmax attention — true
    "guided backprop" (zeroing negative gradients at every activation)
    has minimal effect compared to CNNs.  We instead compute plain
    input × gradient saliency, which is well-defined and informative
    for transformer architectures.

    Pass guided=True to enable the original guided-BP hook (zeroes
    negative input gradients at GELU/ReLU layers).  Leave it False
    (default) for plain input-gradient saliency.
    """

    def __init__(self, model: ViTWithHead, guided: bool = False):
        self.model = model
        self._hooks: list = []
        if guided:
            self._patch_activations()

    def _patch_activations(self):
        def _guided_hook(module, grad_in, grad_out):
            return (F.relu(grad_in[0]),)

        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.GELU)):
                h = module.register_full_backward_hook(_guided_hook)
                self._hooks.append(h)

    @torch.enable_grad()
    def __call__(
        self,
        img_tensor: torch.Tensor,  # (1, 3, 224, 224)
        class_idx: int,
    ) -> np.ndarray:  # (224, 224) saliency in [0, 1]

        self.model.eval()
        img = img_tensor.detach().requires_grad_(True)

        logits = self.model(img)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # input × gradient saliency (signed cancellation removed by abs)
        saliency = (img.grad.data[0] * img.detach()[0]).abs()  # (3, 224, 224)
        saliency = saliency.max(dim=0).values  # (224, 224)
        saliency = saliency.cpu().numpy()
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )
        return saliency

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def guided_gradcam(
    combined_model: ViTWithHead,
    img_tensor: torch.Tensor,
    class_idx: int,
) -> dict:
    """
    Compute Grad-CAM, input-gradient saliency, and their element-wise product
    (Guided Grad-CAM) using a single ViTWithHead model.

    Returns dict with keys: 'gradcam', 'guided_bp', 'guided_gradcam'.
    All values are (224, 224) float32 numpy arrays in [0, 1].

    A single combined model is used so that:
      - hooks are registered once (no interference),
      - gradients flow through the same forward path for both methods.
    """
    cam_extractor = GradCAM(combined_model)
    gbp_extractor = GuidedBackprop(combined_model, guided=False)

    cam = cam_extractor(img_tensor, class_idx)
    # Remove Grad-CAM hooks before running GuidedBackprop to avoid
    # the activation/gradient buffers being overwritten mid-backward.
    cam_extractor.remove_hooks()

    gbp = gbp_extractor(img_tensor, class_idx)
    gbp_extractor.remove_hooks()

    ggc = cam * gbp
    ggc = (ggc - ggc.min()) / (ggc.max() - ggc.min() + 1e-8)

    return {"gradcam": cam, "guided_bp": gbp, "guided_gradcam": ggc}


def explain_pipeline(
    df,
    test_idx,
    label_map,
    head_checkpoint,
    n_samples=10,
    device="cpu",
    stain_normalizer=None,
):
    """Generate Guided Grad-CAM visualizations for n_samples test images."""

    GRADCAM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx2label = {v: k for k, v in label_map.items()}

    # --- Build backbone (feature extractor only) ---
    backbone = torchvision.models.vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT, image_size=224
    )
    # Freeze backbone weights: no parameter updates, but gradients still
    # flow backward through the encoder for Grad-CAM (requires_grad on
    # the *input* tensor drives the backward pass, not the parameters).
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.to(device)

    # --- Load classifier head ---
    ckpt = torch.load(head_checkpoint, map_location=device)
    head = ClassifierHead(768, len(label_map)).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    # --- Combine into a single differentiable model ---
    combined = ViTWithHead(backbone, head).to(device)
    combined.eval()

    for i, idx in enumerate(test_idx[:n_samples]):
        row = df.iloc[idx]
        pil = Image.open(row["path"]).convert("RGB")
        if stain_normalizer:
            pil = stain_normalizer(pil)

        img_t = det_transform(pil).unsqueeze(0).to(device)
        label_idx = int(row["label"])

        maps = guided_gradcam(combined, img_t, label_idx)

        save_path = GRADCAM_OUT_DIR / f"sample_{i:04d}_cls{label_idx}.png"
        save_explanation(pil, maps, idx2label.get(label_idx, str(label_idx)), save_path)
        print(f"=> Saved: {save_path}")


def save_explanation(
    original_img: Image.Image, maps: dict, label: str, save_path: Path
):
    """Save a 4-panel PNG: original | Grad-CAM | Guided BP | Guided Grad-CAM."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")

    for ax, (key, title) in zip(
        axes[1:],
        [
            ("gradcam", "Grad-CAM"),
            ("guided_bp", "Input × Gradient"),
            ("guided_gradcam", "Guided Grad-CAM"),
        ],
    ):
        colored = cm.jet(maps[key])[:, :, :3]
        img = original_img.resize((224, 224))
        overlay = 0.5 * np.array(img) / 255.0 + 0.5 * colored
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(title)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Class: {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class TestGradCAM(unittest.TestCase):
    def test_split(self):
        train_idx, valid_idx, test_idx, label_map = load_dataset(
            LUNG_METADATA_FILE, LUNG_IMAGES_DIR, LUNG_LOADED_FILE
        )

        df = pd.read_csv(LUNG_LOADED_FILE, encoding="utf-8")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = Path(LUNG_MODEL_DIR, "LUNG_ClassifierHead_199_EPOCHS.pth.tar")

        if os.path.exists(ckpt):
            explain_pipeline(
                df,
                test_idx,
                label_map,
                ckpt,
                n_samples=5,
                device=device,
                stain_normalizer=None,
            )
        else:
            print(f"=> ERROR: Model head checkpoint {ckpt} not found")


if __name__ == "__main__":
    unittest.main()
