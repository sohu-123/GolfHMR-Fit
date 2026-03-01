# GolfHMR-Fit: Physics-Constrained Fine-Tuning of Foundation Models for 3D Golf Swing Mesh Recovery

<p align="left">
<a href="#"><img src='https://img.shields.io/badge/Paper-PDF-4A90E2?logo=adobeacrobatreader&logoColor=white' alt='Paper'></a>
<a href="#"><img src='https://img.shields.io/badge/Project_Page-Website-9B72F0?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href="#"><img src='https://img.shields.io/badge/🤗_Hugging_Face-Dataset-F59500?logoColor=white' alt='Dataset'></a>
<a href="#"><img src='https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white' alt='Code'></a>
</p>

**Authors:** Anonymous Author, Anonymous Co-Author  
**Organization:** Anonymous University

---

## Overview

**GolfHMR-Fit** is a targeted fine‑tuning framework designed to enhance monocular 3D human mesh recovery (HMR) for the challenging domain of golf swing analysis. Built upon the foundation model [3D‑SAM‑Body](https://github.com/facebookresearch/sam-3d-body), our approach introduces three key innovations to address the critical issues of hand separation artifacts, occlusion sensitivity, and temporal jitter inherent in general‑purpose HMR methods when applied to high‑speed sports.

### Key Contributions

1. **Physics‑Constrained Priors**  
   - *Geometric grip constraint* – a double‑sided barrier loss that penalizes hand separation beyond biomechanically plausible limits (0.05–0.15 m), eliminating the “surrender” artifact without requiring explicit club detection.  
   - *Kinematic smoothness regularizer* – a second‑order temporal loss that suppresses high‑frequency jitter while preserving the natural acceleration of the swing, ensuring temporally coherent hand trajectories.

2. **Human‑in‑the‑Loop Synthetic Prompting**  
   Generates high‑fidelity 3D hand ground truth for the most challenging frames (e.g., severe occlusion, motion blur) via expert refinement in Blender and 2D reprojection validation (IoU > 95%). These pseudo‑keypoints are then used to supervise fine‑tuning with L1 loss.

3. **GolfSwing3D Dataset**  
   A new benchmark dataset comprising **32 video sequences** (12,868 frames) of golf swings captured from a down‑the‑line viewpoint at 30 fps, with dense 3D hand annotations validated through a human‑in‑the‑loop pipeline (≈18% frames manually refined).

Extensive experiments demonstrate that GolfHMR‑Fit reduces hand‑specific MPJPE by **38.2%** compared to the strong baseline 3D‑SAM‑Body, while maintaining competitive full‑body accuracy and achieving the highest temporal smoothness (0.94).

---

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (recommended) or CPU
- 8GB+ GPU memory (for inference)

### Setup Environment

```bash
# Clone repository (replace with your actual URL)
git clone https://github.com/YOUR_USERNAME/GolfHMR-Fit.git
cd GolfHMR-Fit

# Create conda environment
conda create -n golfhmr python=3.11 -y
conda activate golfhmr

# Install PyTorch (adjust for your CUDA version)
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt


python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/golfhmr-fit/model.ckpt \
    --mhr_path ./checkpoints/golfhmr-fit/assets/mhr_model.pt

python eval.py \
    --image_folder ./data/golfswing3d/test/images \
    --checkpoint_path ./checkpoints/golfhmr-fit/model.ckpt \
    --mhr_path ./checkpoints/golfhmr-fit/assets/mhr_model.pt \
    --output_dir ./eval_results

python train_finetune.py \
    --config configs/finetune_golf.yaml \
    --train_data ./data/golfswing3d/train \
    --val_data ./data/golfswing3d/val

GolfHMR-Fit/
├── golfhmr_fit/               # Core model code
│   ├── __init__.py
│   ├── build_models.py        # Model loading / building
│   ├── estimator.py           # Main inference class
│   ├── losses/                # Physics losses (grip, temporal, prompt)
│   │   ├── grip_loss.py
│   │   ├── temporal_loss.py
│   │   └── prompt_loss.py
│   ├── data/                  # Data loading & preprocessing
│   │   └── dataset.py
│   ├── utils/                 # Utility functions
│   └── visualization/         # Rendering & overlay tools
├── tools/                      # Auxiliary tools
│   ├── vis_utils.py           # Visualization utilities
│   ├── build_detector.py      # Human/hand detector
│   └── blender_refine.py      # Blender refinement helper
├── configs/                    # Configuration files
├── scripts/                    # Training & evaluation scripts
│   ├── train_finetune.py
│   └── eval.py
├── tests/                       # Unit tests
├── demo.py                       # Simple inference script
├── eval.py                       # Evaluation script
├── train_finetune.py             # Fine‑tuning entry point
├── INSTALL.md                    # Detailed installation guide
├── LICENSE
├── README.md                     # This file
└── requirements.txt              # Python dependencies



@article{anonymous2026golfhmrfit,
  title={GolfHMR-Fit: Physics-Constrained Fine-Tuning of Foundation Models for 3D Golf Swing Mesh Recovery},
  author={Anonymous Author and Anonymous Co-Author},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
