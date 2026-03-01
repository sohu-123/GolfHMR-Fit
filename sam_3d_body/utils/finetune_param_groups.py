# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Hand fine-tuning parameter groups for SAM 3D Body.

Strategy (e.g. for golf grip / hand-focused tasks):
- Prompt Embedding (semantic + position): fine-tune @ 1e-4
- Prompt Encoder (MLP projecting 2D -> token): fine-tune @ 1e-4
- Hand Decoder: fine-tune @ 1e-4
- MHR Prediction Head (hand MLP): fine-tune @ 1e-4
- Image Encoder top (last 5 layers): light fine-tune @ 1e-6
- Image Encoder bottom: freeze
- Body Decoder: freeze
"""

from typing import List, Optional, Tuple

import torch.nn as nn


# -----------------------------------------------------------------------------
# 1) Where each "layer" lives in the model (SAM3DBody)
# -----------------------------------------------------------------------------
#
# - Prompt Embedding (语义+位置编码):
#   - sam_3d_body/models/decoders/prompt_encoder.py
#     - PromptEncoder.pe_layer (PositionEmbeddingRandom, position encoding)
#     - PromptEncoder.point_embeddings, not_a_point_embed, invalid_point_embed (semantic)
#   - sam_3d_body/models/meta_arch/sam3d_body.py
#     - keypoint_embedding_hand, keypoint3d_embedding_hand (hand path semantic tokens)
#     - hand_pe_layer (PositionEmbeddingRandom for hand decoder path)
#
# - Prompt Encoder (独立 MLP，2D 点 -> token):
#   - sam3d_body.py: prompt_to_token (Linear: backbone.embed_dims -> DECODER.DIM)
#   - sam3d_body.py: keypoint_feat_linear_hand, keypoint_posemb_linear_hand,
#                    keypoint3d_posemb_linear_hand (hand path projection layers)
#
# - Hand Decoder:
#   - sam3d_body.py: decoder_hand (PromptableDecoder, full transformer)
#
# - MHR Prediction Head (MLP):
#   - sam_3d_body/models/heads/mhr_head.py: MHRHead.proj (FFN, output npose)
#   - sam3d_body.py: head_pose_hand (hand-only MHR head)
#
# - Image Encoder:
#   - sam3d_body.py: backbone (e.g. Dinov3Backbone)
#   - sam_3d_body/models/backbones/dinov3.py: self.encoder (ViT), encoder.blocks[i]
#
# - Body Decoder:
#   - sam3d_body.py: decoder (PromptableDecoder, body path)
# -----------------------------------------------------------------------------


def _collect_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def get_backbone_top_layer_indices(model: nn.Module, num_top_layers: int = 5) -> Optional[Tuple[int, int]]:
    """
    Get (start_idx, end_idx) for the last `num_top_layers` blocks of the image encoder.
    Returns None if backbone does not support layer-wise indexing (e.g. not DINOv3).
    """
    if not hasattr(model, "backbone"):
        return None
    backbone = model.backbone
    if not hasattr(backbone, "encoder"):
        return None
    encoder = backbone.encoder
    if not hasattr(encoder, "n_blocks"):
        return None
    n_blocks = encoder.n_blocks
    if n_blocks < num_top_layers:
        return 0, n_blocks
    start = n_blocks - num_top_layers
    return start, n_blocks


def get_hand_finetune_param_groups(
    model: nn.Module,
    lr_prompt: float = 1e-4,
    lr_encoder_top: float = 1e-6,
    num_encoder_top_layers: int = 5,
    freeze_body_decoder: bool = True,
    freeze_encoder_bottom: bool = True,
) -> List[dict]:
    """
    Build parameter groups for hand-focused fine-tuning.

    Strategy:
        - Prompt embedding + Prompt encoder (incl. prompt_to_token, hand kp projections): lr_prompt
        - Hand decoder: lr_prompt
        - MHR hand head (head_pose_hand): lr_prompt
        - Image encoder top (last N layers): lr_encoder_top
        - Image encoder bottom: frozen (lr 0 or not in optimizer)
        - Body decoder: frozen

    Returns:
        List of {"params": [...], "lr": float, "name": str} for optimizer.
    """
    param_groups: List[dict] = []
    model = model.module if hasattr(model, "module") else model

    # Freeze body decoder and (below) encoder bottom; then add trainable groups.
    if freeze_body_decoder and hasattr(model, "decoder"):
        _set_requires_grad(model.decoder, False)

    # ---- 1) Prompt Embedding (语义+位置) + Prompt Encoder (MLP) @ lr_prompt ----
    prompt_params = []
    if hasattr(model, "prompt_encoder"):
        for p in model.prompt_encoder.parameters():
            p.requires_grad = True
            prompt_params.append(p)
    if hasattr(model, "prompt_to_token"):
        for p in model.prompt_to_token.parameters():
            p.requires_grad = True
            prompt_params.append(p)
    # Hand path: embeddings, position, and projection (2D/3D -> token); init/prev token projection
    for name in (
        "keypoint_embedding_hand",
        "keypoint3d_embedding_hand",
        "hand_pe_layer",
        "keypoint_feat_linear_hand",
        "keypoint_posemb_linear_hand",
        "keypoint3d_posemb_linear_hand",
        "init_to_token_mhr_hand",
        "prev_to_token_mhr_hand",
    ):
        if hasattr(model, name):
            m = getattr(model, name)
            for p in m.parameters():
                p.requires_grad = True
                prompt_params.append(p)
    if prompt_params:
        param_groups.append({"params": prompt_params, "lr": lr_prompt, "name": "prompt_embedding_and_encoder"})

    # ---- 2) Hand Decoder @ lr_prompt ----
    if hasattr(model, "decoder_hand"):
        _set_requires_grad(model.decoder_hand, True)
        hand_dec_params = _collect_params(model.decoder_hand)
        if hand_dec_params:
            param_groups.append({"params": hand_dec_params, "lr": lr_prompt, "name": "hand_decoder"})

    # ---- 3) MHR Prediction Head (hand) @ lr_prompt ----
    if hasattr(model, "head_pose_hand"):
        _set_requires_grad(model.head_pose_hand, True)
        mhr_hand_params = _collect_params(model.head_pose_hand)
        if mhr_hand_params:
            param_groups.append({"params": mhr_hand_params, "lr": lr_prompt, "name": "mhr_prediction_head_hand"})

    # ---- 4) Image Encoder: freeze bottom, light fine-tune top ----
    if not hasattr(model, "backbone"):
        return param_groups
    backbone = model.backbone
    if freeze_encoder_bottom and hasattr(backbone, "encoder"):
        encoder = backbone.encoder
        # Freeze entire encoder first; then unfreeze top layers
        _set_requires_grad(backbone, False)
        indices = get_backbone_top_layer_indices(model, num_encoder_top_layers)
        if indices is not None and hasattr(encoder, "blocks"):
            start, end = indices
            top_params = []
            for i in range(start, end):
                block = encoder.blocks[i]
                for p in block.parameters():
                    p.requires_grad = True
                    top_params.append(p)
            # Also allow patch_embed / pos_embed / norm to be trainable if you want; here we only unfreeze top blocks
            if top_params:
                param_groups.append({"params": top_params, "lr": lr_encoder_top, "name": "image_encoder_top"})
    else:
        # No layer-wise split: either backbone stays frozen or all trainable (not recommended for this strategy)
        pass

    return param_groups


def apply_hand_finetune_strategy(
    model: nn.Module,
    lr_prompt: float = 1e-4,
    lr_encoder_top: float = 1e-6,
    num_encoder_top_layers: int = 5,
) -> List[dict]:
    """
    Apply the full hand fine-tuning strategy (freeze body decoder + encoder bottom,
    set requires_grad and collect param groups). Returns param groups for optimizer.
    """
    return get_hand_finetune_param_groups(
        model,
        lr_prompt=lr_prompt,
        lr_encoder_top=lr_encoder_top,
        num_encoder_top_layers=num_encoder_top_layers,
        freeze_body_decoder=True,
        freeze_encoder_bottom=True,
    )
