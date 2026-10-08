"""SAM 3D Body prompt pipeline: encode (keypoint, mask) prompts and run them
through a cross-attention transformer decoder over (token, image) pairs.

Both adapted from the SAM-style prompt path (Meta, Apache 2.0):
https://github.com/facebookresearch/segment-anything
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from comfy.ldm.cascade.common import LayerNorm2d_op
from comfy.ops import cast_to_input
from comfy.ldm.sam3.sam import PositionEmbeddingRandom

from .transformer import TransformerDecoderLayer


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_body_joints: int,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_body_joints = num_body_joints

        # Keypoint prompts
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList(
            [operations.Embedding(1, embed_dim, device=device, dtype=dtype) for _ in range(self.num_body_joints)]
        )
        self.not_a_point_embed = operations.Embedding(1, embed_dim, device=device, dtype=dtype)
        self.invalid_point_embed = operations.Embedding(1, embed_dim, device=device, dtype=dtype)

        # Mask prompt: 5-stage 2x2 strided conv downscaling to embed_dim.
        LN2d = LayerNorm2d_op(operations)
        mask_in_chans = 256
        self.mask_downscaling = nn.Sequential(
            operations.Conv2d(1, mask_in_chans // 64, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans // 64, device=device, dtype=dtype),
            nn.GELU(),
            operations.Conv2d(mask_in_chans // 64, mask_in_chans // 16, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans // 16, device=device, dtype=dtype),
            nn.GELU(),
            operations.Conv2d(mask_in_chans // 16, mask_in_chans // 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans // 4, device=device, dtype=dtype),
            nn.GELU(),
            operations.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans, device=device, dtype=dtype),
            nn.GELU(),
            operations.Conv2d(mask_in_chans, embed_dim, kernel_size=1, device=device, dtype=dtype),
        )
        # Trained values for the gating conv and no_mask_embed are loaded from the state dict
        self.no_mask_embed = operations.Embedding(1, embed_dim, device=device, dtype=dtype)

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        """Positional encoding over the image-embedding grid; (1, C, H, W)."""
        return self.pe_layer(size)

    def _embed_keypoints(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Embeds point prompts.
        Assuming points have been normalized to [0, 1].

        Output shape [B, N, C], mask shape [B, N]
        """
        assert points.min() >= 0 and points.max() <= 1
        # PE compute in fp32 for precision (sin/cos of large coords), then cast back to the embedding weight dtype
        weight_dtype = self.invalid_point_embed.weight.dtype
        point_embedding = self.pe_layer._encode(points.to(torch.float)).to(weight_dtype)

        # One gather over the stacked joint table.
        joint_w = cast_to_input(torch.cat([e.weight for e in self.point_embeddings], dim=0), point_embedding, copy=False)
        idx = labels.long().clamp(0, self.num_body_joints - 1)
        is_joint = ((labels >= 0) & (labels < self.num_body_joints)).unsqueeze(-1)
        point_embedding = point_embedding + joint_w[idx] * is_joint.to(point_embedding.dtype)

        # -2/-1 zero the PE first, so the embedding replaces it outright.
        invalid_w = cast_to_input(self.invalid_point_embed.weight, point_embedding, copy=False)
        not_a_point_w = cast_to_input(self.not_a_point_embed.weight, point_embedding, copy=False)
        point_embedding = torch.where((labels == -2).unsqueeze(-1), invalid_w, point_embedding)
        point_embedding = torch.where((labels == -1).unsqueeze(-1), not_a_point_w, point_embedding)

        point_mask = labels > -2
        return point_embedding, point_mask

    def _get_batch_size(self, keypoints: Optional[torch.Tensor], boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) -> int:
        if keypoints is not None:
            return keypoints.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          keypoints (torchTensor or none): point coordinates and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(keypoints, boxes, masks)

        ref = keypoints if keypoints is not None else boxes if boxes is not None else masks
        device = ref.device if ref is not None else self.point_embeddings[0].weight.device
        weight_dtype = self.invalid_point_embed.weight.dtype
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device, dtype=weight_dtype)
        sparse_masks = torch.empty((bs, 0), device=device)
        if keypoints is not None:
            coords = keypoints[:, :, :2]
            labels = keypoints[:, :, -1]
            point_embeddings, point_mask = self._embed_keypoints(coords, labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            sparse_masks = torch.cat([sparse_masks, point_mask], dim=1)

        return sparse_embeddings, sparse_masks

    def get_mask_embeddings(self, masks: torch.Tensor, bs: int = 1, size: Tuple[int, int] = (16, 16)) -> torch.Tensor:
        """Embeds mask inputs. Caller casts both outputs to its working dtype."""
        no_mask_embeddings = cast_to_input(self.no_mask_embed.weight, masks).reshape(1, -1, 1, 1).expand(bs, -1, size[0], size[1])
        mask_embeddings = self.mask_downscaling(masks)
        return mask_embeddings, no_mask_embeddings


class PromptableDecoder(nn.Module):
    """Cross-attention transformer decoder over (token, image) pairs."""

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        repeat_pe: bool = False,
        do_interm_preds: bool = False,
        keypoint_token_update: bool = False,
        device=None, dtype=None, operations=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                token_dims=dims,
                context_dims=context_dims,
                num_heads=num_heads,
                head_dims=head_dims,
                mlp_dims=mlp_dims,
                repeat_pe=repeat_pe,
                skip_first_pe=(i == 0),
                device=device,
                dtype=dtype,
                operations=operations,
            )
            for i in range(depth)
        )

        self.norm_final = operations.LayerNorm(dims, eps=1e-6, device=device, dtype=dtype)
        self.do_interm_preds = do_interm_preds
        self.keypoint_token_update = keypoint_token_update

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
    ):
        """
        Args:
            token_embedding: [B, N, C]
            image_embedding: [B, C, H, W] -- flattened to [B, HW, C] inline
        """
        # Channels-last for the transformer.
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        if image_augment is not None:
            image_augment = image_augment.flatten(2).permute(0, 2, 1)
        if hand_embeddings is not None:
            hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
            hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
            if len(hand_augment) == 1:
                # inflate batch dimension
                assert len(hand_augment.shape) == 3
                hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        all_pose_outputs = [] if self.do_interm_preds else None
        if self.do_interm_preds:
            assert token_to_pose_output_fn is not None

        layer_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            if hand_embeddings is None:
                token_embedding, image_embedding = layer(
                    token_embedding, image_embedding,
                    token_augment, image_augment, token_mask,
                )
            else:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    torch.cat([image_embedding, hand_embeddings], dim=1),
                    token_augment,
                    torch.cat([image_augment, hand_augment], dim=1),
                    token_mask,
                )
                image_embedding = image_embedding[:, : image_augment.shape[1]]

            if self.do_interm_preds and layer_idx < len(self.layers) - 1:
                curr = token_to_pose_output_fn(
                    self.norm_final(token_embedding),
                    prev_pose_output=all_pose_outputs[-1] if all_pose_outputs else None,
                    layer_idx=layer_idx,
                )
                all_pose_outputs.append(curr)
                if self.keypoint_token_update:
                    assert keypoint_token_update_fn is not None
                    token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                        token_embedding, token_augment, curr, layer_idx,
                    )

        out = self.norm_final(token_embedding)
        if self.do_interm_preds:
            curr = token_to_pose_output_fn(
                out,
                prev_pose_output=all_pose_outputs[-1] if all_pose_outputs else None,
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr)
            return out, all_pose_outputs
        return out
