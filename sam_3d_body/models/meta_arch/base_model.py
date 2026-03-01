# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Define an abstract base model for consistent format input / processing / output."""

from abc import abstractmethod
from functools import partial
from typing import Dict, Optional

import torch
from yacs.config import CfgNode

from ..optim.fp16_utils import convert_module_to_f16, convert_to_fp16_safe

from .base_lightning_module import BaseLightningModule


class BaseModel(BaseLightningModule):
    def __init__(self, cfg: Optional[CfgNode], **kwargs):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self._initialze_model(**kwargs)

        # Initialize attributes for image-based batch format
        self._max_num_person = None
        self._person_valid = None

    def training_step(self, batch, batch_idx):
        """Default training step that includes optional golf-specific hand constraints.

        Notes:
          - This default is intentionally lightweight: it computes an optional
            base keypoint loss (L2 if batch provides 'keypoints_3d'), and
            adds the three hand constraints when batch.get('is_golf', False).
          - Users may override this in task-specific trainers.
        """
        import torch.nn.functional as F
        from sam_3d_body.utils.hand_constraints import (
            get_hand_idxs_from_cfg,
            get_hand_lambdas_from_cfg,
        )

        # Forward - body decoder
        self._initialize_batch(batch)
        pose_output = self.forward_step(batch, decoder_type="body")
        if "mhr" not in pose_output:
            # Nothing to compute
            loss = 0.0
            self.log("train/loss", loss, prog_bar=True)
            return loss

        mhr_out = pose_output["mhr"]
        joint_global_rots = mhr_out.get("joint_global_rots", None)
        pred_kps_3d = mhr_out.get("pred_keypoints_3d", None)

        # Base loss (if GT available)
        loss = 0.0
        if pred_kps_3d is not None and "keypoints_3d" in batch:
            gt_kps = batch["keypoints_3d"].float()
            # Ensure same shape
            if gt_kps.shape == pred_kps_3d.shape:
                loss = F.mse_loss(pred_kps_3d, gt_kps)

        # Apply golf-specific constraints
        is_golf = batch.get("is_golf", False)
        if is_golf and joint_global_rots is not None:
            idxs = get_hand_idxs_from_cfg(self.cfg)
            lambdas = get_hand_lambdas_from_cfg(self.cfg)

            # Access MHR head helper (if present)
            if hasattr(self, "head_pose") and hasattr(self.head_pose, "compute_hand_constraints"):
                constraints = self.head_pose.compute_hand_constraints(
                    joint_global_rots=joint_global_rots,
                    joints_3d=pred_kps_3d if pred_kps_3d is not None else None,
                    idxs=idxs,
                    max_angle_rad=getattr(self.cfg.TRAIN, "HAND_MAX_ANGLE_RAD", 1.2)
                    if self.cfg is not None and hasattr(self.cfg, "TRAIN")
                    else 1.2,
                )

                loss += lambdas["lambda_hand_rot"] * constraints["loss_bilateral"]
                loss += lambdas["lambda_hand_axis"] * constraints["loss_axis"]
                loss += lambdas["lambda_hand_wrist"] * constraints["loss_wrist"]

        # Log losses
        if hasattr(self, "_log_metric"):
            self._log_metric("train/loss", float(loss), step=None)

        # Lightning expects a tensor when training with optimizers
        if not isinstance(loss, float):
            return loss
        else:
            return torch.tensor(loss, dtype=torch.float32)

    def configure_optimizers(self):
        """Provide optimizer. Uses hand fine-tune param groups when TRAIN.HAND_FINETUNE_STRATEGY is True."""
        import torch

        use_hand_finetune = (
            self.cfg is not None
            and hasattr(self.cfg, "TRAIN")
            and self.cfg.TRAIN.get("HAND_FINETUNE_STRATEGY", False)
        )
        if use_hand_finetune:
            from sam_3d_body.utils.finetune_param_groups import apply_hand_finetune_strategy

            lr_prompt = self.cfg.TRAIN.get("LR_PROMPT", 1e-4)
            lr_encoder_top = self.cfg.TRAIN.get("LR_ENCODER_TOP", 1e-6)
            num_encoder_top_layers = self.cfg.TRAIN.get("NUM_ENCODER_TOP_LAYERS", 5)
            param_groups = apply_hand_finetune_strategy(
                self,
                lr_prompt=lr_prompt,
                lr_encoder_top=lr_encoder_top,
                num_encoder_top_layers=num_encoder_top_layers,
            )
            optimizer = torch.optim.AdamW(param_groups)
            return optimizer
        lr = None
        if self.cfg is not None and hasattr(self.cfg, "TRAIN"):
            lr = getattr(self.cfg.TRAIN, "LR", None)
        if lr is None:
            lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    @abstractmethod
    def _initialze_model(self, **kwargs) -> None:
        pass

    def data_preprocess(
        self,
        inputs: torch.Tensor,
        crop_width: bool = False,
        is_full: bool = False,  # whether for full_branch
        crop_hand: int = 0,
    ) -> torch.Tensor:
        image_mean = self.image_mean if not is_full else self.full_image_mean
        image_std = self.image_std if not is_full else self.full_image_std

        if inputs.max() > 1 and image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and image_mean.max() > 1:
            inputs = inputs * 255.0
        batch_inputs = (inputs - image_mean) / image_std

        if crop_width:
            if crop_hand > 0:
                batch_inputs = batch_inputs[:, :, :, crop_hand:-crop_hand]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                batch_inputs = batch_inputs[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                batch_inputs = batch_inputs[:, :, :, 64:-64]
            else:
                raise Exception

        return batch_inputs

    def _initialize_batch(self, batch: Dict) -> None:
        # Check whether the input batch is with format
        # [batch_size, num_person, ...]
        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
            self._person_valid = self._flatten_person(batch["person_valid"]) > 0
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0
            self._person_valid = None

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._max_num_person:
            # Merge person crops to batch dimension
            shape = x.shape
            x = x.view(self._batch_size * self._max_num_person, *shape[2:])
        return x

    def _unflatten_person(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self._max_num_person:
            x = x.view(self._batch_size, self._max_num_person, *shape[1:])
        return x

    def _get_valid(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._person_valid is not None:
            x = x[self._person_valid]
        return x

    def _full_to_crop(
        self, batch: Dict, pred_keypoints_2d: torch.Tensor
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        affine_trans = self._flatten_person(batch["affine_trans"]).to(
            pred_keypoints_2d_cropped
        )
        img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def _cam_full_to_crop(
        self, batch: Dict, pred_cam_t: torch.Tensor, focal_length: torch.Tensor = None
    ) -> torch.Tensor:
        """Revert the camera translation from full to crop image space"""
        num_person = batch["img"].shape[1]
        cam_int = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, num_person, -1, -1).contiguous()
        )
        bbox_center = self._flatten_person(batch["bbox_center"])
        bbox_size = self._flatten_person(batch["bbox_scale"])[:, 0]
        img_size = self._flatten_person(batch["ori_img_size"])
        input_size = self._flatten_person(batch["img_size"])[:, 0]

        tx, ty, tz = pred_cam_t[:, 0], pred_cam_t[:, 1], pred_cam_t[:, 2]
        if focal_length is None:
            focal_length = cam_int[:, 0, 0]
        bs = 2 * focal_length / (tz + 1e-8)

        cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
        cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        crop_cam_t = torch.stack(
            [tx - cx, ty - cy, tz * bbox_size / input_size], dim=-1
        )
        return crop_cam_t

    def convert_to_fp16(self) -> torch.dtype:
        """
        Convert the torso of the model to float16.
        """
        fp16_type = (
            torch.float16
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16"
            else torch.bfloat16
        )

        if hasattr(self, "backbone"):
            self._set_fp16(self.backbone, fp16_type)
        if hasattr(self, "full_encoder"):
            self._set_fp16(self.full_encoder, fp16_type)

        if hasattr(self.backbone, "lhand_pos_embed"):
            self.backbone.lhand_pos_embed.data = self.backbone.lhand_pos_embed.data.to(
                fp16_type
            )

        if hasattr(self.backbone, "rhand_pos_embed"):
            self.backbone.rhand_pos_embed.data = self.backbone.rhand_pos_embed.data.to(
                fp16_type
            )

        return fp16_type

    def _set_fp16(self, module, fp16_type):
        if hasattr(module, "pos_embed"):
            module.apply(partial(convert_module_to_f16, dtype=fp16_type))
            module.pos_embed.data = module.pos_embed.data.to(fp16_type)
        elif hasattr(module.encoder, "rope_embed"):
            # DINOv3
            module.encoder.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.encoder.rope_embed = module.encoder.rope_embed.to(fp16_type)
        else:
            # DINOv2
            module.encoder.pos_embed.data = module.encoder.pos_embed.data.to(fp16_type)
