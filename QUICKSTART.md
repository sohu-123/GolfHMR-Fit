# Quick Start Guide

## 立即开始使用 SAM 3D Body

### 最小化安装（5分钟）

```bash
# 1. 克隆项目
git clone https://github.com/YOUR_USERNAME/sam-3d-body.git
cd sam-3d-body

# 2. 创建虚拟环境
conda create -n sam3d python=3.11 -y
conda activate sam3d

# 3. 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装依赖
pip install -r requirements.txt

# 5. 下载模型
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir ./checkpoints/sam-3d-body-dinov3
```

### 运行推理

```bash
# 使用文件夹中的图像
python demo.py \
    --image_folder /path/to/images \
    --output_folder ./output \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

### Python 快速示例

```python
import cv2
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together

# 初始化
model, cfg = load_sam_3d_body(
    "./checkpoints/sam-3d-body-dinov3/model.ckpt",
    device="cuda",
    mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
estimator = SAM3DBodyEstimator(model, cfg)

# 推理
img_bgr = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
outputs = estimator.process_one_image(img_rgb)

# 保存结果
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

### 评估

```bash
python eval.py \
    --image_folder /path/to/test/images \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --output_dir ./eval_results
```

## 下一步

- 详见 [README_NEW.md](README_NEW.md)（完整文档）
- 查看 [INSTALL.md](INSTALL.md)（详细安装）
- 运行 [notebook/demo_human.ipynb](notebook/demo_human.ipynb)（交互式演示）

## 常见问题

**Q: 需要GPU吗？**
A: 推荐使用GPU。CPU 推理速度较慢。

**Q: 模型多大？**
A: DINOv3-H+ 约 840MB，ViT-H 约 631MB。

**Q: 支持哪些输入？**
A: JPG、PNG、BMP 等常见图像格式。

更多问题见 [README_NEW.md](README_NEW.md) 中的 "Advanced Usage"。
