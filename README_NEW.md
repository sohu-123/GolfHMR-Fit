# SAM 3D Body: Robust Full-Body Human Mesh Recovery

<p align="left">
<a href="https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"><img src='https://img.shields.io/badge/Meta_AI-Paper-4A90E2?logo=meta&logoColor=white' alt='Paper'></a>
<a href="https://ai.meta.com/blog/sam-3d/"><img src='https://img.shields.io/badge/Project_Page-Blog-9B72F0?logo=googledocs&logoColor=white' alt='Blog'></a>
<a href="https://huggingface.co/datasets/facebook/sam-3d-body-dataset"><img src='https://img.shields.io/badge/🤗_Hugging_Face-Dataset-F59500?logoColor=white' alt='Dataset'></a>
<a href="https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d"><img src='https://img.shields.io/badge/🤸_Playground-Live_Demo-E85D5D?logoColor=white' alt='Live Demo'></a>
</p>

**Authors:** Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt Feiszli, Jitendra Malik, Piotr Dollar, Kris Kitani

**Organization:** Meta Superintelligence Labs

## Overview

**SAM 3D Body (3DB)** is a promptable model for single-image full-body 3D human mesh recovery (HMR). It demonstrates state-of-the-art performance with strong generalization across diverse real-world scenarios. The model estimates human pose for the body, feet, and hands based on the [Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR), a parametric mesh representation that decouples skeletal structure and surface shape.

### Key Features
- **Promptable Inference**: Supports optional 2D keypoints and mask prompts for user-guided 3D reconstruction
- **Full-Body Modeling**: Captures body, feet, and hand poses simultaneously
- **Strong Generalization**: Robust performance across diverse in-the-wild conditions
- **Efficient Architecture**: Encoder-decoder design with DINOv3 and ViT-H backbones
- **Multi-Modal Training**: Trained on high-quality annotations from a multi-stage pipeline

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (recommended) or CPU
- 8GB+ GPU memory (for inference)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sam-3d-body.git
cd sam-3d-body

# Create conda environment
conda create -n sam3d_body python=3.11 -y
conda activate sam3d_body

# Install PyTorch (adjust for your CUDA version)
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core webdataset roma joblib seaborn wandb appdirs ffmpeg tensorboard huggingface_hub

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# Install additional dependencies (optional)
# For MoGe FOV estimation
pip install git+https://github.com/microsoft/MoGe.git

# For SAM3 detector
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
```

### Download Model Checkpoints

Download from Hugging Face:
```bash
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir ./checkpoints/sam-3d-body-dinov3
# or
huggingface-cli download facebook/sam-3d-body-vith --local-dir ./checkpoints/sam-3d-body-vith
```

## Quick Start

### Basic Inference

```bash
# Using DINOv3 backbone with default detector
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

### With SAM3 Detector (Recommended for Fine Details)

```bash
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name sam3
```

### Python API

```python
import cv2
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together

# Initialize model
device = "cuda"  # or "cpu"
model, cfg = load_sam_3d_body(
    "./checkpoints/sam-3d-body-dinov3/model.ckpt",
    device=device,
    mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)

# Create estimator
estimator = SAM3DBodyEstimator(model, cfg)

# Load image
img_bgr = cv2.imread("path/to/image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Run inference
outputs = estimator.process_one_image(img_rgb)

# Visualize results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

## Evaluation

Evaluate on benchmark datasets:

```bash
python eval.py \
    --image_folder <path_to_images> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --output_dir ./eval_results
```

**Output Metrics:**
- **MPJPE**: Mean Per Joint Position Error (mm)
- **PA-MPJPE**: Procrustes-Aligned MPJPE (mm)
- **PVE**: Per Vertex Error (mm)

### Benchmark Results

| Backbone | 3DPW (MPJPE) | EMDB (MPJPE) | RICH (PVE) | COCO (PCK@0.05) | LSPet (PCK@0.05) | FreiHAND (PA-MPJPE) |
|----------|------------|-----------|---------|------------|------------|-----------|
| DINOv3-H+ (840M) | 54.8 | 61.7 | 60.3 | 86.5 | 68.0 | 5.5 |
| ViT-H (631M) | 54.8 | 62.9 | 61.7 | 86.8 | 68.9 | 5.5 |

## Project Structure

```
sam-3d-body/
├── sam_3d_body/              # Core model code
│   ├── build_models.py
│   ├── sam_3d_body_estimator.py
│   ├── models/               # Model architectures
│   ├── data/                 # Data processing
│   ├── utils/                # Utilities
│   └── visualization/        # Visualization tools
├── tools/                    # Auxiliary tools
│   ├── vis_utils.py         # Visualization utilities
│   ├── build_detector.py    # Human detection
│   ├── build_sam.py         # SAM segmentation
│   └── build_fov_estimator.py
├── tests/                    # Unit tests
├── notebook/                 # Jupyter notebooks
│   └── demo_human.ipynb
├── demo.py                   # Inference script
├── eval.py                   # Evaluation script
├── INSTALL.md                # Installation guide
├── LICENSE
├── README.md
└── requirements.txt
```

## Advanced Usage

### With Custom Keypoint Prompts

```python
# Provide 2D keypoints to guide reconstruction
outputs = estimator.process_one_image(
    img_rgb,
    keypoints_2d=keypoints_array,  # (N_keypoints, 2)
    keypoint_mask=mask_array        # (N_keypoints,)
)
```

### With Segmentation Masks

```python
# Use segmentation mask for better localization
outputs = estimator.process_one_image(
    img_rgb,
    mask=segmentation_mask  # (H, W) binary mask
)
```

## Notebooks

For comprehensive examples with visualizations:
- [demo_human.ipynb](notebook/demo_human.ipynb) - Full demo with body, hands, and feet

## Dataset

Download the SAM 3D Body dataset:
```bash
huggingface-cli download facebook/sam-3d-body-dataset --local-dir ./data/sam-3d-body
```

See [data/README.md](data/README.md) for dataset details and preprocessing instructions.

## Training & Finetuning

For fine-tuning on custom data:
```bash
python train_finetune_hand_with_constraints.py \
    --config <config_path> \
    --train_data <train_data_path> \
    --val_data <val_data_path>
```

Refer to [docs/FINETUNE_STRATEGY.md](docs/FINETUNE_STRATEGY.md) for detailed fine-tuning guidelines.

## License

This project is licensed under the SAM License. See [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Related Work

- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) - Object mesh reconstruction
- [Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR) - Parametric mesh representation
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)

## Acknowledgments

This project builds upon contributions from Meta Superintelligence Labs and collaborators including researchers from UC Berkeley and MPI-IS. Special thanks to all dataset annotators and community contributors.

## Contact & Support

For issues, feature requests, and discussions:
- GitHub Issues: [Project Issues](https://github.com/YOUR_USERNAME/sam-3d-body/issues)
- Email: [contact info]

---

**Last Updated:** November 2025
