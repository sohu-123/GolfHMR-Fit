# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Evaluation script for SAM 3D Body
Computes metrics: MPJPE (Mean Per Joint Position Error), PVE (Per Vertex Error), PA-MPJPE (Procrustes-aligned MPJPE)
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam_3d_body import load_sam_3d_body


def compute_similarity_transform(S1, S2):
    """
    Compute the similarity transform between two point sets.
    Adapted from https://github.com/akanazawa/hmr
    """
    assert S1.shape == S2.shape
    m = S1.shape[0]

    # Compute mean
    mean_S1 = S1.mean(axis=0)
    mean_S2 = S2.mean(axis=0)

    # Center the point sets
    S1_c = S1 - mean_S1
    S2_c = S2 - mean_S2

    # Compute scale
    scale_S1 = np.sqrt((S1_c**2).sum()) / m
    scale_S2 = np.sqrt((S2_c**2).sum()) / m

    S1_n = S1_c / scale_S1
    S2_n = S2_c / scale_S2

    # Compute rotation using SVD
    H = S1_n.T @ S2_n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return S1_n @ R.T * scale_S2 + mean_S2


def compute_mpjpe(pred, target):
    """Compute Mean Per Joint Position Error."""
    assert pred.shape == target.shape
    return np.mean(np.linalg.norm(pred - target, axis=1))


def compute_pa_mpjpe(pred, target):
    """Compute Procrustes Aligned MPJPE."""
    pred_aligned = compute_similarity_transform(pred, target)
    return compute_mpjpe(pred_aligned, target)


def compute_pve(pred_verts, target_verts):
    """Compute Per Vertex Error."""
    assert pred_verts.shape == target_verts.shape
    return np.mean(np.linalg.norm(pred_verts - target_verts, axis=1))


class SAM3DBodyEvaluator:
    def __init__(self, checkpoint_path, mhr_path, device="cuda"):
        """Initialize evaluator with model and MHR."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path, device=self.device, mhr_path=mhr_path
        )
        self.model.eval()

    @torch.no_grad()
    def evaluate_batch(self, images, gt_keypoints_3d=None, gt_vertices=None):
        """
        Evaluate predictions on a batch.
        
        Args:
            images: List of image paths or numpy arrays
            gt_keypoints_3d: Ground truth 3D keypoints (B, J, 3)
            gt_vertices: Ground truth 3D vertices (B, V, 3)
            
        Returns:
            dict: Metrics including MPJPE, PA-MPJPE, PVE
        """
        metrics = {
            "mpjpe": [],
            "pa_mpjpe": [],
            "pve": [],
        }

        for i, img in enumerate(tqdm(images, desc="Evaluating")):
            # Load image
            if isinstance(img, str):
                img_bgr = cv2.imread(img)
                if img_bgr is None:
                    print(f"Warning: Could not load image {img}")
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            # Run inference
            outputs = self.model.process_one_image(img_rgb)

            # Compute metrics if ground truth available
            if gt_keypoints_3d is not None and i < len(gt_keypoints_3d):
                pred_kpts = outputs.get("joints_3d", np.zeros((25, 3)))
                gt_kpts = gt_keypoints_3d[i]
                
                if pred_kpts.shape == gt_kpts.shape:
                    mpjpe = compute_mpjpe(pred_kpts, gt_kpts)
                    pa_mpjpe = compute_pa_mpjpe(pred_kpts, gt_kpts)
                    metrics["mpjpe"].append(mpjpe)
                    metrics["pa_mpjpe"].append(pa_mpjpe)

            if gt_vertices is not None and i < len(gt_vertices):
                pred_verts = outputs.get("vertices", np.zeros((10475, 3)))
                gt_verts = gt_vertices[i]
                
                if pred_verts.shape == gt_verts.shape:
                    pve = compute_pve(pred_verts, gt_verts)
                    metrics["pve"].append(pve)

        # Compute mean metrics
        results = {}
        for key, values in metrics.items():
            if len(values) > 0:
                results[f"{key}_mean"] = float(np.mean(values))
                results[f"{key}_std"] = float(np.std(values))
                
        return results


def main(args):
    """Main evaluation function."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = SAM3DBodyEvaluator(
        args.checkpoint_path, args.mhr_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Get image list
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_folder = Path(args.image_folder)
    images = [
        str(p) for p in image_folder.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]
    images.sort()

    print(f"Found {len(images)} images")

    # Run evaluation
    results = evaluator.evaluate_batch(images)

    # Save results
    output_path = os.path.join(args.output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Evaluation Results:")
    print(f"{'='*50}")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*50}\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM 3D Body")
    parser.add_argument(
        "--image_folder", type=str, required=True, help="Path to folder containing images"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mhr_path", type=str, required=True, help="Path to MHR model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_output", help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    args = parser.parse_args()
    main(args)
