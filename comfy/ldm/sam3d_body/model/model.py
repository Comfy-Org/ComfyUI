from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
from comfy.ops import cast_to_input
from comfy.ldm.sam3.sam import PositionEmbeddingRandom

from comfy.image_encoders.dino3 import DINOV3_VITH_CONFIG, DINOv3ViTModel
from .prompt import PromptEncoder, PromptableDecoder
from ..mhr.mhr_head import MHRHead
from ..mhr.mhr_rig import MHRRig
from ..mhr.mhr_utils import fix_wrist_euler, rotation_angle_difference
from .transformer import MLP
from .camera_modules import CameraEncoder, PerspectiveHead
from comfy_extras.mediapipe.face_landmarker import FaceLandmarker
from ..utils import bbox_xyxy2cs, fix_aspect_ratio, get_warp_matrices, warp_affine_batched, euler_to_rotmat, rotmat_to_euler

# Architecture constants for the released `dinov3-h+` SAM 3D Body checkpoint.
IMAGE_SIZE = (512, 512)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DECODER_DIM = 1024
DECODER_DEPTH = 6
DECODER_HEADS = 8
DECODER_DIM_HEAD = 64
DECODER_MLP_DIM = 1024
MHR_MLP_DEPTH = 2
CAMERA_MLP_DEPTH = 2
CAMERA_DEFAULT_SCALE_FACTOR_HAND = 10.0
N_KEYPOINTS = 70  # mhr70


class SAM3DBody(nn.Module):
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        # Per-batch state populated by `_initialize_batch`.
        self._max_num_person = None

        self.register_buffer("image_mean", torch.tensor(IMAGE_MEAN).view(-1, 1, 1), False)
        self.register_buffer("image_std", torch.tensor(IMAGE_STD).view(-1, 1, 1), False)

        self.image_size = IMAGE_SIZE
        self.backbone = DINOv3ViTModel(DINOV3_VITH_CONFIG, dtype=dtype, device=device, operations=operations, use_mask_token=False)
        embed_dims = self.backbone.embed_dims

        # MHR rig shared between body + hand pose heads via a non-registered
        # Python ref, so state_dict has one top-level `mhr.*` key tree (not
        # duplicated under `head_pose.mhr.*` AND `head_pose_hand.mhr.*`).
        self.mhr = MHRRig(device=device)

        head_kwargs = dict(
            input_dim=DECODER_DIM,
            mlp_depth=MHR_MLP_DEPTH,
            mhr_rig=self.mhr,
            mlp_channel_div_factor=1,
            device=device, dtype=dtype, operations=operations,
        )
        self.head_pose = MHRHead(**head_kwargs)
        self.init_pose = operations.Embedding(1, self.head_pose.npose, device=device, dtype=dtype)

        self.head_pose_hand = MHRHead(enable_hand_model=True, **head_kwargs)
        self.init_pose_hand = operations.Embedding(
            1, self.head_pose_hand.npose, device=device, dtype=dtype
        )

        camera_kwargs = dict(
            input_dim=DECODER_DIM,
            img_size=IMAGE_SIZE,
            mlp_depth=CAMERA_MLP_DEPTH,
            mlp_channel_div_factor=1,
            device=device, dtype=dtype, operations=operations,
        )
        self.head_camera = PerspectiveHead(**camera_kwargs)
        self.init_camera = operations.Embedding(1, self.head_camera.ncam, device=device, dtype=dtype)

        self.head_camera_hand = PerspectiveHead(default_scale_factor=CAMERA_DEFAULT_SCALE_FACTOR_HAND, **camera_kwargs)
        self.init_camera_hand = operations.Embedding(1, self.head_camera_hand.ncam, device=device, dtype=dtype)

        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        linear_kwargs = dict(device=device, dtype=dtype)
        self.init_to_token_mhr = operations.Linear(init_dim, DECODER_DIM, **linear_kwargs)
        self.prev_to_token_mhr = operations.Linear(init_dim - cond_dim, DECODER_DIM, **linear_kwargs)
        self.init_to_token_mhr_hand = operations.Linear(init_dim, DECODER_DIM, **linear_kwargs)
        self.prev_to_token_mhr_hand = operations.Linear(init_dim - cond_dim, DECODER_DIM, **linear_kwargs)

        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dims,  # match backbone dims so PE adds directly
            num_body_joints=N_KEYPOINTS,
            device=device, dtype=dtype, operations=operations,
        )
        self.prompt_to_token = operations.Linear(embed_dims, DECODER_DIM, **linear_kwargs)

        decoder_kwargs = dict(
            dims=DECODER_DIM,
            context_dims=embed_dims,
            depth=DECODER_DEPTH,
            num_heads=DECODER_HEADS,
            head_dims=DECODER_DIM_HEAD,
            mlp_dims=DECODER_MLP_DIM,
            repeat_pe=True,
            do_interm_preds=True,
            keypoint_token_update="v2",
            device=device, dtype=dtype, operations=operations,
        )
        self.decoder = PromptableDecoder(**decoder_kwargs)
        self.decoder_hand = PromptableDecoder(**decoder_kwargs)
        self.hand_pe_layer = PositionEmbeddingRandom(embed_dims // 2)

        # Inference-time dtype set by the Loader via model.backbone.to(dtype).
        self.backbone_dtype = torch.float32

        ray_kwargs = dict(
            embed_dim=embed_dims, patch_size=self.backbone.patch_size,
            device=device, dtype=dtype, operations=operations,
        )
        self.ray_cond_emb = CameraEncoder(**ray_kwargs)
        self.ray_cond_emb_hand = CameraEncoder(**ray_kwargs)

        self.keypoint_embedding_idxs = list(range(N_KEYPOINTS))
        self.keypoint_embedding_idxs_hand = list(range(N_KEYPOINTS))
        self.keypoint_embedding = operations.Embedding(N_KEYPOINTS, DECODER_DIM, **linear_kwargs)
        self.keypoint_embedding_hand = operations.Embedding(N_KEYPOINTS, DECODER_DIM, **linear_kwargs)

        self.hand_box_embedding = operations.Embedding(2, DECODER_DIM, **linear_kwargs)
        self.bbox_embed = MLP(
            input_dim=DECODER_DIM, hidden_dim=DECODER_DIM,
            output_dim=4, num_layers=3,
            device=device, dtype=dtype, operations=operations,
        )

        posemb_kwargs = dict(
            hidden_dim=DECODER_DIM, output_dim=DECODER_DIM, num_layers=2,
            device=device, dtype=dtype, operations=operations,
        )
        self.keypoint_posemb_linear = MLP(input_dim=2, **posemb_kwargs)
        self.keypoint_posemb_linear_hand = MLP(input_dim=2, **posemb_kwargs)
        self.keypoint_feat_linear = operations.Linear(embed_dims, DECODER_DIM, **linear_kwargs)
        self.keypoint_feat_linear_hand = operations.Linear(embed_dims, DECODER_DIM, **linear_kwargs)

        self.keypoint3d_embedding_idxs = list(range(N_KEYPOINTS))
        self.keypoint3d_embedding_idxs_hand = list(range(N_KEYPOINTS))
        self.keypoint3d_embedding = operations.Embedding(N_KEYPOINTS, DECODER_DIM, **linear_kwargs)
        self.keypoint3d_embedding_hand = operations.Embedding(N_KEYPOINTS, DECODER_DIM, **linear_kwargs)
        self.keypoint3d_posemb_linear = MLP(input_dim=3, **posemb_kwargs)
        self.keypoint3d_posemb_linear_hand = MLP(input_dim=3, **posemb_kwargs)

        self.face_landmarker = FaceLandmarker(
            device=device, dtype=torch.float32, operations=None,
            detector_variant="both", # short+full, picks whichever found more faces per frame.
        )

    @staticmethod
    def memory_used_forward(crops: int, with_hand_refinement: bool) -> int:
        # CUDA reserved-memory calibration at 512px; the 64-crop hand path keeps about 6% headroom.
        per_crop = 200 if with_hand_refinement else 112
        return (384 + max(1, int(crops)) * per_crop) * 1024 ** 2

    def data_preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        image_mean = cast_to_input(self.image_mean, inputs, copy=False)
        image_std = cast_to_input(self.image_std, inputs, copy=False)
        return (inputs - image_mean) / image_std

    def _initialize_batch(self, batch: Dict) -> None:
        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"
        if self._max_num_person:
            x = x.view(self._batch_size * self._max_num_person, *x.shape[2:])
        return x

    def _set_active_branch(self, kind: str) -> None:
        """Route subsequent calls through the body or hand decoder by switching
        which batch indices are active."""
        n = self._batch_size * self._max_num_person
        all_idx = list(range(n))
        if kind == "body":
            self.body_batch_idx, self.hand_batch_idx = all_idx, []
        elif kind == "hand":
            self.body_batch_idx, self.hand_batch_idx = [], all_idx
        else:
            raise ValueError(f"Invalid branch kind: {kind!r}")

    @staticmethod
    def _concat_hand_batches(a: Dict, b: Dict) -> Dict:
        """Merge two prepare_batch dicts along dim 0 for a single hand pass.
        Tensors cat, lists extend, scalars/metadata taken from `a`."""
        out = {}
        for k, va in a.items():
            vb = b.get(k)
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                out[k] = torch.cat([va, vb], dim=0)
            elif isinstance(va, list) and isinstance(vb, list):
                out[k] = va + vb
            else:
                out[k] = va
        return out

    @staticmethod
    def _split_hand_output(batched: Dict, n_left: int) -> Tuple[Dict, Dict]:
        """Inverse of `_concat_hand_batches`. Only `mhr_hand` needs splitting;
        condition_info / image_embeddings aren't consumed downstream."""
        batched_mhr = batched["mhr_hand"]
        lhand_mhr: Dict[str, Any] = {}
        rhand_mhr: Dict[str, Any] = {}
        for k, v in batched_mhr.items():
            if isinstance(v, torch.Tensor):
                lhand_mhr[k] = v[:n_left]
                rhand_mhr[k] = v[n_left:]
            else:
                # numpy `faces`, `pred_pose_rotmat=None`, etc. -- shared.
                lhand_mhr[k] = v
                rhand_mhr[k] = v
        return (
            {"mhr": None, "mhr_hand": lhand_mhr},
            {"mhr": None, "mhr_hand": rhand_mhr},
        )

    def _prepare_hand_batches_gpu(
        self,
        img,
        left_xyxy: torch.Tensor,
        right_xyxy: torch.Tensor,
        cam_int: torch.Tensor,
        is_multi_image: bool,
    ) -> Tuple[Dict, Dict]:
        """Build batch_lhand + batch_rhand directly on GPU. Bit-exact match
        for the CPU `prepare_batch` × 2 path, with the source uploaded once
        and both warps issued through one batched grid_sample."""

        device = comfy.model_management.get_torch_device()
        if is_multi_image:
            assert isinstance(img, list)
            n = len(img)
            src_t = torch.stack(list(img), dim=0)
        else:
            n = int(left_xyxy.shape[0])
            src_t = img.unsqueeze(0).expand(n, -1, -1, -1)

        H_out, W_out = int(self.image_size[0]), int(self.image_size[1])
        bbox_padding = 0.9  # matches transform_hand
        aspect = 0.75

        def _meta(boxes_xyxy):
            centers, scales = bbox_xyxy2cs(boxes_xyxy, padding=bbox_padding)
            scales = fix_aspect_ratio(scales, aspect)
            scales = fix_aspect_ratio(scales, W_out / H_out)
            mats = get_warp_matrices(centers, scales, (H_out, W_out))
            return centers, scales, mats

        l_centers, l_scales, l_mats = _meta(left_xyxy)
        r_centers, r_scales, r_mats = _meta(right_xyxy)

        src_t = src_t.to(device=device, dtype=torch.float32, non_blocking=True).permute(0, 3, 1, 2)

        warped_l = warp_affine_batched(torch.flip(src_t, dims=[3]), l_mats, (H_out, W_out))
        warped_r = warp_affine_batched(src_t, r_mats, (H_out, W_out))
        # floor -> /255 matches the per-item uint8 round-trip path.
        l_img = (torch.floor(warped_l).clamp_(0.0, 255.0) / 255.0).contiguous()
        r_img = (torch.floor(warped_r).clamp_(0.0, 255.0) / 255.0).contiguous()

        # All-zero mask + score 0 (matches prepare_batch's masks=None path).
        zero_mask = torch.zeros((n, 1, H_out, W_out), dtype=torch.float32, device=device)
        zero_mask_score = torch.zeros((n,), dtype=torch.float32, device=device)
        person_valid = torch.ones((1, n), dtype=torch.float32, device=device)
        img_size = torch.tensor([W_out, H_out], dtype=torch.float32, device=device).expand(n, 2).contiguous()
        cam_int_dev = cam_int.to(device=device, dtype=torch.float32)

        def _build(centers_t, scales_t, mats_t, img_t, boxes_xyxy):
            return {
                "img": img_t.unsqueeze(0),  # (1, N, 3, H_out, W_out)
                "img_size": img_size.unsqueeze(0),
                "bbox_center": centers_t.unsqueeze(0),
                "bbox_scale": scales_t.unsqueeze(0),
                "bbox": torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=device).unsqueeze(0),
                "affine_trans": mats_t.unsqueeze(0),
                "mask": zero_mask.unsqueeze(0),  # (1, N, 1, H_out, W_out)
                "mask_score": zero_mask_score.unsqueeze(0),  # (1, N)
                "person_valid": person_valid,  # (1, N) shared OK
                "cam_int": cam_int_dev,
            }

        return (
            _build(l_centers, l_scales, l_mats, l_img, left_xyxy),
            _build(r_centers, r_scales, r_mats, r_img, right_xyxy),
        )

    # Forward path

    def _get_decoder_condition(self, batch: Dict) -> torch.Tensor:
        """CLIFF-style condition: ((cx-img_cx)/f, (cy-img_cy)/f, b/f), all in [-1, 1]."""
        num_person = batch["img"].shape[1]
        cx, cy = torch.chunk(self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1)
        b = self._flatten_person(batch["bbox_scale"])[:, [0]]

        cam_int_per_person = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, num_person, -1, -1).contiguous()
        )
        focal_length = cam_int_per_person[:, 0, 0]
        full_img_cxy = cam_int_per_person[:, [0, 1], [2, 2]]
        condition_info = torch.cat(
            [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1,
        )
        condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(-1)
        condition_info[:, 2] = condition_info[:, 2] / focal_length
        return condition_info.type(batch["img"].dtype)

    @staticmethod
    def _append_token_block(token_embeddings, token_augment, embedding_weight, batch_size):
        """Append a token block from `embedding_weight` (+ zero-block in
        token_augment). Returns (token_embeddings, token_augment, start_idx)."""
        start_idx = token_embeddings.shape[1]
        block = cast_to_input(embedding_weight, token_embeddings)[None, :, :].repeat(batch_size, 1, 1)
        token_embeddings = torch.cat([token_embeddings, block], dim=1)
        token_augment = torch.cat([token_augment, torch.zeros_like(block)], dim=1)
        return token_embeddings, token_augment, start_idx

    def forward_decoder(
        self,
        branch: str,
        image_embeddings: torch.Tensor,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """`branch` selects body or hand decoder + paired attribute set; rest
        of the pipeline is shared.

        image_embeddings: (B, C, H, W) backbone features.
        keypoints:        (B, N, 3) prompts as (x, y in [0, 1], label).
                          label: 0..K = joint, -1 = incorrect, -2 = invalid.
        prev_estimate:    (B, 1, C) previous estimate for pose refinement.
        condition_info:   (B, c) extra condition concatenated to input tokens.
        """
        if branch == "body":
            init_pose_emb = self.init_pose
            init_camera_emb = self.init_camera
            init_to_token = self.init_to_token_mhr
            prev_to_token = self.prev_to_token_mhr
            ray_cond_emb = self.ray_cond_emb
            ray_cond_key = "ray_cond"
            head_pose = self.head_pose
            head_camera = self.head_camera
            keypoint_embedding = self.keypoint_embedding
            keypoint3d_embedding = self.keypoint3d_embedding
            decoder = self.decoder
            batch_idx = self.body_batch_idx
            # Body shares the prompt encoder's PE.
            image_augment_fn = self.prompt_encoder.get_dense_pe
        elif branch == "hand":
            init_pose_emb = self.init_pose_hand
            init_camera_emb = self.init_camera_hand
            init_to_token = self.init_to_token_mhr_hand
            prev_to_token = self.prev_to_token_mhr_hand
            ray_cond_emb = self.ray_cond_emb_hand
            ray_cond_key = "ray_cond_hand"
            head_pose = self.head_pose_hand
            head_camera = self.head_camera_hand
            keypoint_embedding = self.keypoint_embedding_hand
            keypoint3d_embedding = self.keypoint3d_embedding_hand
            decoder = self.decoder_hand
            batch_idx = self.hand_batch_idx
            # Hand decoder has its own PE layer (not the prompt encoder's).
            image_augment_fn = self.hand_pe_layer
        else:
            raise ValueError(f"Invalid branch: {branch!r}")

        batch_size = image_embeddings.shape[0]

        init_pose = cast_to_input(init_pose_emb.weight, image_embeddings).expand(batch_size, -1).unsqueeze(1)
        init_camera = cast_to_input(init_camera_emb.weight, image_embeddings).expand(batch_size, -1).unsqueeze(1)
        init_estimate = torch.cat([init_pose, init_camera], dim=-1)  # B x 1 x (404 + 3)

        init_input = torch.cat([condition_info.view(batch_size, 1, -1), init_estimate], dim=-1)
        token_embeddings = init_to_token(init_input).view(batch_size, 1, -1)
        num_pose_token = token_embeddings.shape[1]  # always 1

        image_augment, token_augment, token_mask = None, None, None
        if keypoints is not None:
            if prev_estimate is None:
                prev_estimate = init_estimate
            prev_embeddings = prev_to_token(prev_estimate).view(batch_size, 1, -1)

            # PE generated in fp32; cast back to decoder dtype.
            image_augment = image_augment_fn(image_embeddings.shape[-2:]).to(image_embeddings)

            # ray_cond is fp32 from get_ray_condition; cast so CameraEncoder's
            # internal cat doesn't silently promote everything back to fp32.
            image_embeddings = ray_cond_emb(
                image_embeddings, batch[ray_cond_key].type(image_embeddings.dtype),
            )

            # Keypoints start as [0, 0, -2]. Labels select the embedding
            # weight (special for -2, -1, then per joint).
            prompt_embeddings, _ = self.prompt_encoder(keypoints=keypoints)
            prompt_embeddings = self.prompt_to_token(prompt_embeddings)

            # Pin dtypes so a silent fp16→fp32 promotion in any branch
            # (init/prev/prompt) doesn't break the index_put assigns below.
            token_embeddings = torch.cat(
                [token_embeddings, prev_embeddings, prompt_embeddings], dim=1,
            ).to(image_embeddings.dtype)
            prev_embeddings = prev_embeddings.to(image_embeddings.dtype)
            prompt_embeddings = prompt_embeddings.to(image_embeddings.dtype)

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1):] = prompt_embeddings

            token_embeddings, token_augment, hand_det_emb_start_idx = self._append_token_block(
                token_embeddings, token_augment, self.hand_box_embedding.weight, batch_size,
            )
            token_embeddings, token_augment, kps_emb_start_idx = self._append_token_block(
                token_embeddings, token_augment, keypoint_embedding.weight, batch_size,
            )
            token_embeddings, token_augment, kps3d_emb_start_idx = self._append_token_block(
                token_embeddings, token_augment, keypoint3d_embedding.weight, batch_size,
            )

        last_layer_idx = len(decoder.layers) - 1
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            pose_token = tokens[:, 0]
            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)
            # Suppress vertices on non-final layers — kp-token updates only
            # need keypoints, so we skip the 18439-vertex perspective projection.
            is_intermediate = layer_idx != last_layer_idx
            pose_output = head_pose(pose_token, prev_pose, intermediate=is_intermediate)
            pose_output["pred_cam"] = head_camera(pose_token, prev_camera)
            pose_output = self.camera_project(pose_output, batch, branch=branch)
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], batch_idx,
            )
            return pose_output

        def keypoint_token_update_fn_comb(*args):
            args = self._keypoint_token_update(branch, kps_emb_start_idx, image_embeddings, *args)
            args = self._keypoint3d_token_update(branch, kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        return (
            pose_token[:, hand_det_emb_start_idx:hand_det_emb_start_idx + 2],
            pose_output,
        )


    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        x_mask = x_mask.to(image_embeddings.dtype)

        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1).to(image_embeddings.dtype)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full-image kp coords → crop-normalized [-0.5, 0.5]."""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict, branch: str = "body") -> Dict:
        """Project 3D keypoints (+ optional vertices) to 2D. `branch` selects
        the body or hand attribute set + batch slice."""
        head_camera = self.head_camera_hand if branch == "hand" else self.head_camera
        batch_idx = self.hand_batch_idx if branch == "hand" else self.body_batch_idx
        pred_cam = pose_output["pred_cam"]

        # Hoist the shared bbox/intrinsics slice so we don't recompute the
        # expand+contiguous for the vertices branch.
        bbox_center = self._flatten_person(batch["bbox_center"])[batch_idx]
        bbox_scale = self._flatten_person(batch["bbox_scale"])[batch_idx, 0]
        cam_int = self._flatten_person(
            batch["cam_int"]
            .unsqueeze(1)
            .expand(-1, batch["img"].shape[1], -1, -1)
            .contiguous()
        )[batch_idx]

        def _project(points_3d):
            return head_camera.perspective_projection(
                points_3d, pred_cam, bbox_center, bbox_scale, cam_int,
            )

        cam_out = _project(pose_output["pred_keypoints_3d"])
        if pose_output.get("pred_vertices") is not None:
            pose_output["pred_keypoints_2d_verts"] = _project(
                pose_output["pred_vertices"]
            )["pred_keypoints_2d"]

        pose_output.update(cam_out)
        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        affine = batch["affine_trans"]
        xs, ys = torch.meshgrid(
            torch.arange(W, device=affine.device, dtype=affine.dtype),
            torch.arange(H, device=affine.device, dtype=affine.dtype),
            indexing="xy",
        )
        meshgrid_xy = torch.stack([xs, ys], dim=-1)[None, None].expand(B, N, -1, -1, -1)
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """One pose-decoder pass over the crop batch (body and/or hand)."""
        batch_size, num_person = batch["img"].shape[:2]

        x = self.data_preprocess(self._flatten_person(batch["img"]))

        ray_cond = self._flatten_person(self.get_ray_condition(batch))
        if len(self.body_batch_idx):
            batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
        if len(self.hand_batch_idx):
            batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
        ray_cond = None

        image_embeddings = self.backbone.forward_features(x.type(self.backbone_dtype))
        # bf16 mantissa too lossy for the heads — promote back. fp16 survives.
        if self.backbone_dtype != torch.float16:
            image_embeddings = image_embeddings.type(x.dtype)

        image_embeddings = image_embeddings + self._get_mask_prompt(batch, image_embeddings)

        # condition_info is fp32 from `_get_decoder_condition`; align to
        # decoder dtype so the downstream cat doesn't auto-promote.
        condition_info = self._get_decoder_condition(batch).type(image_embeddings.dtype)

        # Seed prompt: all-invalid keypoints (label = -2).
        keypoints_prompt = torch.zeros(
            (batch_size * num_person, 1, 3), dtype=batch["img"].dtype, device=batch["img"].device,
        )
        keypoints_prompt[:, :, -1] = -2

        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                "body",
                image_embeddings[self.body_batch_idx],
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]
        if len(self.hand_batch_idx):
            tokens_output_hand, pose_output_hand = self.forward_decoder(
                "hand",
                image_embeddings[self.hand_batch_idx],
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
            )
            pose_output_hand = pose_output_hand[-1]

        output = {
            "mhr": pose_output,
            "mhr_hand": pose_output_hand,
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }
        # hand_box is (x1, y1, w, h) ∈ [0, 1]. Body path promotes to fp32 to
        # match the head-MLP external contract (_get_hand_box would .float() anyway).
        if len(self.body_batch_idx):
            output["mhr"]["hand_box"] = self.bbox_embed(tokens_output).sigmoid().float()
        if len(self.hand_batch_idx):
            output["mhr_hand"]["hand_box"] = self.bbox_embed(tokens_output_hand).sigmoid()

        return output

    def forward_step(self, batch: Dict, decoder_type: str = "body") -> Tuple[Dict, Dict]:
        self._set_active_branch(decoder_type)
        return self.forward_pose_branch(batch)

    def run_inference(
        self,
        img,
        batch: Dict,
        inference_type: str = "full",
        thresh_wrist_angle=1.4,
    ):
        """3DB inference. inference_type: 'full' (body + hand-refined),
        'body' (body decoder only), 'hand' (hand decoder only)."""

        self._initialize_batch(batch)
        is_multi_image = isinstance(img, list)
        ref_img = img[0] if is_multi_image else img
        height, width = ref_img.shape[:2]
        cam_int = batch["cam_int"].clone()

        if inference_type == "body":
            return self.forward_step(batch, decoder_type="body")
        if inference_type == "hand":
            return self.forward_step(batch, decoder_type="hand")
        if inference_type != "full":
            raise ValueError(f"Invalid inference type: {inference_type!r}")

        # 1. Body decoder pass.
        pose_output = self.forward_step(batch, decoder_type="body")
        left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)
        ori_local_wrist_rotmat = euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3)),
        )

        # 2. Hand re-run. Flip the left box's x so it indexes into the
        # to-be-flipped source image (the flip itself happens on GPU inside
        # _prepare_hand_batches_gpu — no CPU copy of the frames needed).
        tmp = left_xyxy.clone()
        left_xyxy[:, 0] = width - tmp[:, 2] - 1
        left_xyxy[:, 2] = width - tmp[:, 0] - 1

        batch_lhand, batch_rhand = self._prepare_hand_batches_gpu(
            img, left_xyxy, right_xyxy, cam_int.clone(), is_multi_image,
        )
        # Concat lhand+rhand along dim 0 so backbone+decoder run once on
        # (2, num_person, ...) — saves one full DINOv3 ViT-H+ pass.
        batch_hands = self._concat_hand_batches(batch_lhand, batch_rhand)
        saved_batch_state = (self._batch_size, self._max_num_person)
        self._initialize_batch(batch_hands)
        hands_output = self.forward_step(batch_hands, decoder_type="hand")
        self._batch_size, self._max_num_person = saved_batch_state
        n_left = batch_lhand["img"].shape[0] * batch_lhand["img"].shape[1]
        lhand_output, rhand_output = self._split_hand_output(hands_output, n_left)
        # Free the batched image_embeddings/condition_info (unused downstream);
        # mhr_hand views into the underlying tensors stay alive via l/rhand_output.
        del hands_output, batch_hands

        # Unflip left-hand output. Keep MHR consts as 0-d on-device tensors —
        # `.item()` would force four hard CPU<->GPU syncs in the hot path.
        _lhand_scale = lhand_output["mhr_hand"]["scale"]
        scale_r_hands_mean = self.head_pose.scale_mean[8].to(_lhand_scale)
        scale_l_hands_mean = self.head_pose.scale_mean[9].to(_lhand_scale)
        scale_r_hands_std = self.head_pose.scale_comps[8, 8].to(_lhand_scale)
        scale_l_hands_std = self.head_pose.scale_comps[9, 9].to(_lhand_scale)
        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (scale_r_hands_mean + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8])
            - scale_l_hands_mean
        ) / scale_l_hands_std
        # Right-hand global rotation flipped → used as left.
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = \
            lhand_output["mhr_hand"]["joint_global_rots"][:, 42].clone()
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][:, 54:]
        batch_lhand["bbox_center"][:, :, 0] = width - batch_lhand["bbox_center"][:, :, 0] - 1

        # 3. Validity criteria for replacing body-decoder hand pose.
        # (a) local wrist pose difference: hand vs body wrist rotations
        joint_rotations = pose_output["mhr"]["joint_global_rots"]
        lowarm_joint_rotations = joint_rotations[:, [76, 40]]
        wrist_zero_rot_pose = lowarm_joint_rotations @ cast_to_input(
            self.head_pose.joint_rotation[[77, 41]], lowarm_joint_rotations, copy=False,
        )
        pred_global_wrist_rotmat = torch.stack(
            [lhand_output["mhr_hand"]["joint_global_rots"][:, 78],
             rhand_output["mhr_hand"]["joint_global_rots"][:, 42]],
            dim=1,
        )
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose,
        )
        angle_difference_valid_mask = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat,
        ) < thresh_wrist_angle

        # (b) hand box big enough to give the decoder useful pixels
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack(
            [(batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
             (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1)],
            dim=1,
        )

        # (c) all hand 2D keypoints inside the crop box
        hand_kps2d_thresh = 0.5
        hand_kps2d_valid_mask = torch.stack(
            [lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2)) < hand_kps2d_thresh,
             rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2)) < hand_kps2d_thresh],
            dim=1,
        )

        # (d) hand-decoder wrist close to body-decoder wrist in 2D
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx, kps_left_wrist_idx = 41, 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1
        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_left_wrist_idx]].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(dim=-1) \
            / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(dim=-1) \
            / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack(
            [left_kps_dist < hand_wrist_kps2d_thresh,
             right_kps_dist < hand_wrist_kps2d_thresh],
            dim=1,
        )

        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )

        # Re-prompt body decoder with hand-decoder wrists + body-decoder elbows
        # to get an updated body pose estimation.
        self._set_active_branch("body")

        # right_kps_full / left_kps_full already computed above (unchanged since).
        right_kps_crop = self._full_to_crop(batch, right_kps_full)
        left_kps_crop = self._full_to_crop(batch, left_kps_full)

        kps_right_elbow_idx, kps_left_elbow_idx = 8, 7
        right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_right_elbow_idx]].clone()
        left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_left_elbow_idx]].clone()
        right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
        left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)

        keypoint_prompt = torch.cat(
            [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop], dim=1,
        )
        keypoint_prompt = torch.cat([keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1)
        keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
        keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
        keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
        keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

        if keypoint_prompt.shape[0] > 1:
            invalid_prompt = (
                (keypoint_prompt[..., 0] < -0.5)
                | (keypoint_prompt[..., 0] > 0.5)
                | (keypoint_prompt[..., 1] < -0.5)
                | (keypoint_prompt[..., 1] > 0.5)
                | (~hand_valid_mask[..., [1, 0, 1, 0]])
            ).unsqueeze(-1)
            dummy_prompt = torch.zeros((1, 1, 3), dtype=keypoint_prompt.dtype, device=keypoint_prompt.device)
            dummy_prompt[:, :, -1] = -2
            # Shift [-0.5, 0.5] → [0, 1] for the prompt encoder.
            keypoint_prompt[:, :, :2] = torch.clamp(keypoint_prompt[:, :, :2] + 0.5, 0.0, 1.0)
            keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
        else:
            valid_keypoint = (
                torch.all(
                    (keypoint_prompt[:, :, :2] > -0.5) & (keypoint_prompt[:, :, :2] < 0.5),
                    dim=2,
                )
                & hand_valid_mask[..., [1, 0, 1, 0]]
            ).squeeze()
            keypoint_prompt = keypoint_prompt[:, valid_keypoint]
            keypoint_prompt[:, :, :2] = torch.clamp(keypoint_prompt[:, :, :2] + 0.5, 0.0, 1.0)

        if keypoint_prompt.numel() != 0:
            pose_output, _ = self.run_keypoint_prompt(batch, pose_output, keypoint_prompt)

        # 4. Drop hand pose / scale / shape from the hand decoder into the body output.
        updated_hand_pose = torch.cat(
            [lhand_output["mhr_hand"]["hand"][:, :54],
             rhand_output["mhr_hand"]["hand"][:, 54:]],
            dim=1,
        )
        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2
        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2

        # 5. IK: solve local wrist Euler from the (updated) global wrist rotmat.
        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]
        lowarm_joint_rotations = joint_rotations[:, [76, 40]]
        # joint_rotation is a static buffer at head dtype; cast to MHR's fp32
        # to keep the rotation matmul in fp32.
        wrist_zero_rot_pose = lowarm_joint_rotations @ cast_to_input(
            self.head_pose.joint_rotation[[77, 41]], joint_rotations, copy=False,
        )
        pred_global_wrist_rotmat = torch.stack(
            [lhand_output["mhr_hand"]["joint_global_rots"][:, 78],
             rhand_output["mhr_hand"]["joint_global_rots"][:, 42]],
            dim=1,
        )
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose,
        )
        wrist_xzy = fix_wrist_euler(rotmat_to_euler("XZY", fused_local_wrist_rotmat))

        valid_angle = (
            (rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat) < thresh_wrist_angle)
            & hand_valid_mask
        ).unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][
            :, [41, 43, 42, 31, 33, 32]
        ].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
            updated_body_pose.flatten(1, 2)
        )

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
            dim=1,
        )
        updated_hand_scale = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_hand_scale = torch.where(
            valid_angle.squeeze(-1), updated_hand_scale, hand_scale
        )
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

        # Replace shared shape and scale
        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )

        # Re-run MHR forward with the updated parameters.
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=pose_output["mhr"]["hand"],
            scale_params=pose_output["mhr"]["scale"],
            shape_params=pose_output["mhr"]["shape"],
            expr_params=pose_output["mhr"]["face"],
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        # j3d: 308 → 70 body/hand kps + 238 face landmarks. All four buffers
        # need the same y/z flip so they share a coordinate system.
        j3d_face = j3d[:, 70:].clone()
        j3d = j3d[:, :70]
        verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        j3d_face[..., [1, 2]] *= -1
        jcoords[..., [1, 2]] *= -1
        pose_output["mhr"]["pred_keypoints_3d"] = j3d
        pose_output["mhr"]["pred_face_keypoints_3d"] = j3d_face
        pose_output["mhr"]["pred_vertices"] = verts
        pose_output["mhr"]["pred_joint_coords"] = jcoords
        pose_output["mhr"]["joint_global_rots"] = joint_global_rots
        pose_output["mhr"]["pred_pose_raw"][...] = 0  # invalidated by the IK update
        pose_output["mhr"]["mhr_model_params"] = mhr_model_params

        def _project_kp3d(kp3d: torch.Tensor) -> torch.Tensor:
            proj = kp3d + pose_output["mhr"]["pred_cam_t"][:, None, :]
            proj[:, :, [0, 1]] = proj[:, :, [0, 1]] * pose_output["mhr"]["focal_length"][:, None, None]
            proj[:, :, [0, 1]] = (
                proj[:, :, [0, 1]]
                + proj.new_tensor([width / 2, height / 2])[None, None, :]
                * proj[:, :, [2]]
            )
            proj[:, :, :2] = proj[:, :, :2] / proj[:, :, [2]]
            return proj[:, :, :2]

        pose_output["mhr"]["pred_keypoints_2d"] = _project_kp3d(
            pose_output["mhr"]["pred_keypoints_3d"].clone()
        )
        pose_output["mhr"]["pred_face_keypoints_2d"] = _project_kp3d(
            pose_output["mhr"]["pred_face_keypoints_3d"].clone()
        )

        return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"],
                pose_output["shape"],
                pose_output["scale"],
                pose_output["hand"],
                pose_output["face"],
            ],
            dim=1,
        ).unsqueeze(1)
        prev_estimate = torch.cat(
            [prev_estimate, pose_output["pred_cam"].unsqueeze(1)], dim=-1,
        )

        _, pose_output = self.forward_decoder(
            "body",
            image_embeddings,
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Hand bbox from the detector → full-image coords (xyxy). Stays on
        device throughout."""
        hand_box = pose_output["mhr"]["hand_box"]  # (B, 2, 4) fp32
        pred_left_hand_box = hand_box[:, 0] * self.image_size[0]
        pred_right_hand_box = hand_box[:, 1] * self.image_size[0]

        # Square the boxes (long side wins).
        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = pred_left_hand_box[:, 2:].amax(dim=1, keepdim=True).repeat(1, 2)
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = pred_right_hand_box[:, 2:].amax(dim=1, keepdim=True).repeat(1, 2)

        # Invert the crop's full→crop affine. rot=0 makes it diagonal:
        # divide-by-scale and subtract translation offset.
        affine_trans = batch["affine_trans"][0]
        affine_scale = affine_trans[:, 0, 0]
        affine_offset = affine_trans[:, :2, 2]
        batch["left_scale"] = batch["left_scale"] / affine_scale[:, None]
        batch["right_scale"] = batch["right_scale"] / affine_scale[:, None]
        batch["left_center"] = (batch["left_center"] - affine_offset) / affine_scale[:, None]
        batch["right_center"] = (batch["right_center"] - affine_offset) / affine_scale[:, None]

        left_xyxy = torch.stack(
            [
                batch["left_center"][:, 0] - batch["left_scale"][:, 0] / 2,
                batch["left_center"][:, 1] - batch["left_scale"][:, 1] / 2,
                batch["left_center"][:, 0] + batch["left_scale"][:, 0] / 2,
                batch["left_center"][:, 1] + batch["left_scale"][:, 1] / 2,
            ],
            dim=1,
        )
        right_xyxy = torch.stack(
            [
                batch["right_center"][:, 0] - batch["right_scale"][:, 0] / 2,
                batch["right_center"][:, 1] - batch["right_scale"][:, 1] / 2,
                batch["right_center"][:, 0] + batch["right_scale"][:, 0] / 2,
                batch["right_center"][:, 1] + batch["right_scale"][:, 1] / 2,
            ],
            dim=1,
        )

        return left_xyxy, right_xyxy


    # Shared 2D-keypoint-driven token update. `branch` picks body/hand attrs;
    # rest is identical. Called via keypoint_token_update_fn_comb in
    # forward_decoder.
    def _keypoint_token_update(
        self,
        branch: str,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        if branch == "body":
            decoder_layers = self.decoder.layers
            kp_emb_w = self.keypoint_embedding.weight
            kp_idxs = self.keypoint_embedding_idxs
            posemb_linear = self.keypoint_posemb_linear
            feat_linear = self.keypoint_feat_linear
        else:
            decoder_layers = self.decoder_hand.layers
            kp_emb_w = self.keypoint_embedding_hand.weight
            kp_idxs = self.keypoint_embedding_idxs_hand
            posemb_linear = self.keypoint_posemb_linear_hand
            feat_linear = self.keypoint_feat_linear_hand

        # Last layer's pose output is final — nothing to inject back.
        if layer_idx == len(decoder_layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        token_augment = token_augment.clone()
        num_keypoints = kp_emb_w.shape[0]

        # kp comes from fp32 MHR/cam projection; cast once to decoder dtype
        # so posemb / grid_sample match.
        pred_keypoints_2d_cropped = pose_output["pred_keypoints_2d_cropped"].clone()[:, kp_idxs]
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()[:, kp_idxs]
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped.to(image_embeddings.dtype)

        # Mask out-of-frame OR behind-camera keypoints' contributions.
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            posemb_linear(pred_keypoints_2d_cropped) * (~invalid_mask[:, :, None])
        )

        # Bilinear-sample image features at each kp's projected location.
        # grid_sample wants -1..1; cropped form is -0.5..0.5, so ×2.
        sample_points = pred_keypoints_2d_cropped * 2
        feats = F.grid_sample(
            image_embeddings, sample_points[:, :, None, :],
            mode="bilinear", padding_mode="zeros", align_corners=False,
        ).squeeze(3).permute(0, 2, 1)
        feats = feats * (~invalid_mask[:, :, None])

        token_embeddings = token_embeddings.clone()
        token_embeddings[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] += (
            feat_linear(feats)
        )

        return token_embeddings, token_augment, pose_output, layer_idx

    def _keypoint3d_token_update(
        self,
        branch: str,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        if branch == "body":
            decoder_layers = self.decoder.layers
            kp3d_emb_w = self.keypoint3d_embedding.weight
            kp3d_idxs = self.keypoint3d_embedding_idxs
            posemb_linear = self.keypoint3d_posemb_linear
        else:
            decoder_layers = self.decoder_hand.layers
            kp3d_emb_w = self.keypoint3d_embedding_hand.weight
            kp3d_idxs = self.keypoint3d_embedding_idxs_hand
            posemb_linear = self.keypoint3d_posemb_linear_hand

        if layer_idx == len(decoder_layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = kp3d_emb_w.shape[0]

        # Pelvis-normalize so 3D kps live in subject-centric coords (don't
        # leak global cam translation into the token signal). Cast to decoder
        # dtype before posemb_linear writes back into token_augment.
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()
        pred_keypoints_3d = pred_keypoints_3d - (
            pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
            + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
        ) / 2
        pred_keypoints_3d = pred_keypoints_3d[:, kp3d_idxs].to(token_augment.dtype)

        token_augment = token_augment.clone()
        token_augment[:, kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d, :] = (
            posemb_linear(pred_keypoints_3d)
        )

        return token_embeddings, token_augment, pose_output, layer_idx
