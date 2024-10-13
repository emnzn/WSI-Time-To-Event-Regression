import os

import cv2
import torch
import numpy as np
import pandas as pd

from .preprocess import min_max_scale, merge_patches

def visualize_attention(
    patch_shape: int,
    model_name: str,
    patient_table: pd.DataFrame,
    attention: torch.Tensor,
    save_dir: str,
    patient_id: int
    ):

    if model_name == "attention-mil":
        patient_table["attention"] = attention.squeeze().cpu().numpy()
        patient_table["attention"] = min_max_scale(patient_table["attention"])
        patient_table["attention"] = patient_table["attention"].map(lambda x: np.expand_dims(np.ones((patch_shape, patch_shape)) * x, axis=-1))

        merged_attention = merge_patches(patient_table["attention"].tolist(), patient_table["adjusted_coords"].tolist(), target_patch_size=patch_shape)
        merged_img = merge_patches(patient_table["img"].tolist(), patient_table["adjusted_coords"].tolist(), target_patch_size=patch_shape).astype("uint8")

        patient_table["attention"] = attention.squeeze().cpu().numpy()
        patient_table["attention"] = min_max_scale(patient_table["attention"])
        patient_table["attention"] = patient_table["attention"].map(lambda x: np.expand_dims(np.ones((patch_shape, patch_shape)) * x, axis=-1))

        merged_attention = merge_patches(patient_table["attention"].tolist(), patient_table["adjusted_coords"].tolist(), target_patch_size=patch_shape)
        merged_img = merge_patches(patient_table["img"].tolist(), patient_table["adjusted_coords"].tolist(), target_patch_size=patch_shape).astype("uint8")

        merged_attention_bgr = cv2.applyColorMap((merged_attention * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        merged_img_bgr = cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR)

        alpha = 0.5

        overlay = cv2.addWeighted(merged_img_bgr, 0.5, merged_attention_bgr, alpha, 0)
        merged_attention_bgr[(merged_img == [0, 0, 0]).all(axis=-1)] = [0, 0 ,0]
        overlay[(merged_img == [0, 0, 0]).all(axis=-1)] = [0, 0 ,0]

        cv2.imwrite(os.path.join(save_dir, f"{patient_id}.png"), overlay)