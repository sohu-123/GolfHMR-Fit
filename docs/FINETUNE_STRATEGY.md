# 手部微调策略：各层位置与配置说明

## 1. 各层在代码中的位置

| 模块 | 是否微调 | 学习率 | 代码位置 |
|------|----------|--------|----------|
| **Prompt Embedding（语义+位置编码）** | ✅ 微调 | 1e-4 | `sam_3d_body/models/decoders/prompt_encoder.py`: `PromptEncoder.pe_layer`, `point_embeddings`, `not_a_point_embed`, `invalid_point_embed`；<br>`sam_3d_body/models/meta_arch/sam3d_body.py`: `keypoint_embedding_hand`, `keypoint3d_embedding_hand`, `hand_pe_layer` |
| **Prompt Encoder（独立 MLP）** | ✅ 微调 | 1e-4 | `sam3d_body.py`: `prompt_to_token`；<br>`keypoint_feat_linear_hand`, `keypoint_posemb_linear_hand`, `keypoint3d_posemb_linear_hand`；<br>`init_to_token_mhr_hand`, `prev_to_token_mhr_hand` |
| **Hand Decoder** | ✅ 重点微调 | 1e-4 | `sam3d_body.py`: `decoder_hand`（`PromptableDecoder`） |
| **MHR Prediction Head（MLP）** | ✅ 微调 | 1e-4 | `sam_3d_body/models/heads/mhr_head.py`: `MHRHead.proj`；<br>`sam3d_body.py`: `head_pose_hand`（手部 MHR 头） |
| **Image Encoder 顶层（最后 5 层）** | ⚠️ 轻微微调 | 1e-6 | `sam_3d_body/models/backbones/dinov3.py`: `backbone.encoder.blocks[-5:]`（DINOv3 时） |
| **Image Encoder 底层** | ❌ 冻结 | 0 | `backbone.encoder` 除上述顶层外的部分 |
| **Body Decoder** | ❌ 冻结 | 0 | `sam3d_body.py`: `decoder`（身体路径 decoder） |

## 2. 如何启用该策略

### 方式 A：通过配置（推荐）

在训练用的 `model_config.yaml`（或与 checkpoint 同目录的配置）中增加或修改：

```yaml
TRAIN:
  HAND_FINETUNE_STRATEGY: true   # 启用手部微调策略
  LR_PROMPT: 1.0e-4              # Prompt / Hand Decoder / MHR Head 学习率（默认 1e-4）
  LR_ENCODER_TOP: 1.0e-6         # Image Encoder 顶层学习率（默认 1e-6）
  NUM_ENCODER_TOP_LAYERS: 5      # 微调的 encoder 顶层数量（默认 5）
```

`BaseModel.configure_optimizers()` 会读取这些项并自动构建参数组（只优化上述需微调部分，其余冻结）。

### 方式 B：在脚本中手动构建优化器

不用配置时，可在训练脚本里直接调用工具函数拿到参数组，再创建优化器：

```python
from sam_3d_body.utils.finetune_param_groups import get_hand_finetune_param_groups

# 对已加载的 model（SAM3DBody）应用策略并得到参数组
param_groups = get_hand_finetune_param_groups(
    model,
    lr_prompt=1e-4,
    lr_encoder_top=1e-6,
    num_encoder_top_layers=5,
    freeze_body_decoder=True,
    freeze_encoder_bottom=True,
)
optimizer = torch.optim.AdamW(param_groups)
```

如需更细控制（例如只微调其中几类），可参考 `sam_3d_body/utils/finetune_param_groups.py` 中的 `get_hand_finetune_param_groups`，按模块名筛选参数后自行组装 `param_groups`。

## 3. 相关代码文件

- **参数组与冻结逻辑**: `sam_3d_body/utils/finetune_param_groups.py`  
  - `apply_hand_finetune_strategy(model, ...)`：应用完整策略并返回 param_groups  
  - `get_hand_finetune_param_groups(model, ...)`：可配置冻结/学习率，返回 param_groups  
- **优化器接入**: `sam_3d_body/models/meta_arch/base_model.py` 中的 `configure_optimizers()`，当 `TRAIN.HAND_FINETUNE_STRATEGY` 为 true 时使用上述 param_groups。
