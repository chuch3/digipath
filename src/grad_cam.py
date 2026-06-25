import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import ViT_B_16_Weights

from constant import (
    GRADCAM_OUT_DIR,
    LUNG_IMAGES_DIR,
    LUNG_LOADED_FILE,
    LUNG_METADATA_FILE,
    LUNG_MODEL_DIR,
)
from dataset import load_dataset
from model import ClassifierHead
from transform import det_transform


class ViTWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b._process_input(x)
        cls = b.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = b.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor[:, 1:, :]
    n = result.shape[1]
    side = int(n**0.5)
    result = result.reshape(result.shape[0], side, side, result.shape[2])
    result = result.permute(0, 3, 1, 2)
    return result


def run_gradcam(
    df,
    test_idx,
    label_map,
    head_checkpoint,
    n_samples=10,
    device="cpu",
    stain_normalizer=None,
):
    GRADCAM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    backbone = torchvision.models.vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT, image_size=224
    )
    backbone.to(device)
    backbone.eval()

    ckpt = torch.load(head_checkpoint, map_location=device)
    head = ClassifierHead(768, len(label_map)).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    combined = ViTWithHead(backbone, head).to(device)
    combined.eval()

    target_layer = combined.backbone.encoder.layers[-4]

    cam = GradCAM(
        model=combined,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    )

    for i, idx in enumerate(test_idx[:n_samples]):
        row = df.iloc[idx]
        pil = Image.open(row["path"]).convert("RGB")
        if stain_normalizer:
            pil = stain_normalizer(pil)

        img_t = det_transform(pil).unsqueeze(0).to(device)
        label_idx = int(row["label"])
        targets = [ClassifierOutputTarget(label_idx)]

        grayscale_cam = cam(input_tensor=img_t, targets=targets)[0]

        display_img = pil.resize((224, 224), resample=Image.BILINEAR)
        rgb_img = np.array(display_img).astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        grad_pil = Image.fromarray(visualization)

        save_path = GRADCAM_OUT_DIR / f"gradcam_sample_{i:04d}_cls{label_idx}.png"
        control_path = GRADCAM_OUT_DIR / f"control_sample_{i:04d}_cls{label_idx}.png"

        pil.save(control_path)
        grad_pil.save(save_path)

        print(f"=> Saved: {save_path}")

    # Return latest control sample with Grad-CAM sample and label
    return pil, grad_pil, label_idx


class TestGradCAM(unittest.TestCase):
    def test_split(self):
        train_idx, valid_idx, test_idx, label_map = load_dataset(
            LUNG_METADATA_FILE, LUNG_IMAGES_DIR, LUNG_LOADED_FILE
        )
        df = pd.read_csv(LUNG_LOADED_FILE, encoding="utf-8")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = Path(LUNG_MODEL_DIR, "LUNG_ClassifierHead_199_EPOCHS.pth.tar")
        if os.path.exists(ckpt):
            run_gradcam(
                df,
                test_idx,
                label_map,
                ckpt,
                n_samples=10000,
                device=device,
                stain_normalizer=None,
            )
        else:
            print(f"=> ERROR: Model head checkpoint {ckpt} not found")


if __name__ == "__main__":
    unittest.main()
