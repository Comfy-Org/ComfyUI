# TripoSplat 3D gaussian container. Operates on already-decoded
# tensors and exposes them as render-ready tensors (render_tensors) for the generic SPLAT type.
import torch
import torch.nn.functional as F

import comfy.model_management


class GaussianModel:
    def __init__(self, aabb: list, sh_degree: int = 0, mininum_kernel_size: float = 0.0,
                 scaling_bias: float = 0.01, opacity_bias: float = 0.1,
                 scaling_activation: str = "exp", device=None):
        self.sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

        if scaling_activation == "exp":
            self._scaling_activation = torch.exp
            self._inverse_scaling_activation = torch.log
        elif scaling_activation == "softplus":
            self._scaling_activation = F.softplus
            self._inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self._opacity_activation = torch.sigmoid
        self._inverse_opacity_activation = lambda x: torch.log(x / (1 - x))

        self.scale_bias = self._inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros(4, device=self.device)
        self.rots_bias[0] = 1
        self.opacity_bias_val = self._inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

        self._storage = {}

    def _get_store(self, name):
        return self._storage.get(name)

    def _set_store(self, name, value):
        self._storage[name] = value

    @property
    def _xyz(self):
        return self._get_store("_xyz")
    @_xyz.setter
    def _xyz(self, value):
        if value is None:
            self._set_store("_xyz", None)
            self._set_store("xyz", None)
            return
        self._set_store("_xyz", value)
        self._set_store("xyz", value * self.aabb[None, 3:] + self.aabb[None, :3])

    @property
    def get_xyz(self):
        return self._get_store("xyz")

    @property
    def _features_dc(self):
        return self._get_store("_features_dc")
    @_features_dc.setter
    def _features_dc(self, value):
        self._set_store("_features_dc", value)

    @property
    def _opacity(self):
        return self._get_store("_opacity")
    @_opacity.setter
    def _opacity(self, value):
        if value is None:
            self._set_store("_opacity", None)
            self._set_store("opacity", None)
            return
        self._set_store("_opacity", value)
        self._set_store("opacity", self._opacity_activation(value + self.opacity_bias_val))

    @property
    def get_opacity(self):
        return self._get_store("opacity")

    @property
    def _scaling(self):
        return self._get_store("_scaling")
    @_scaling.setter
    def _scaling(self, value):
        if value is None:
            self._set_store("_scaling", None)
            self._set_store("scaling", None)
            return
        self._set_store("_scaling", value)
        s = self._scaling_activation(value + self.scale_bias)
        s = torch.square(s) + self.mininum_kernel_size ** 2
        self._set_store("scaling", torch.sqrt(s))

    @property
    def get_scaling(self):
        return self._get_store("scaling")

    @property
    def _rotation(self):
        return self._get_store("_rotation")
    @_rotation.setter
    def _rotation(self, value):
        self._set_store("_rotation", value)

    _DEFAULT_TRANSFORM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    def render_tensors(self):
        # Render-ready (activated, world-space) tensors for the generic SPLAT type. The axis transform
        # (a 3x3 rotation, object frame -> viewer Y-up) is baked into positions and rotations.
        # Returns float tensors on the intermediate device: positions (N,3), scales (N,3) linear,
        # rotations (N,4) wxyz, opacities (N,1) in [0,1], sh (N,K,3) coefficients.
        xyz = self.get_xyz.float()
        scaling = self.get_scaling.float()
        opacity = self.get_opacity.float()
        rotation = (self._rotation + self.rots_bias[None, :]).float()
        sh = self._features_dc.float()  # (N, K, 3)
        T = torch.as_tensor(self._DEFAULT_TRANSFORM, dtype=torch.float32, device=xyz.device)
        xyz = xyz @ T.T
        rotation = _matrix_to_quat(torch.matmul(T, _quat_to_matrix(rotation)))
        rotation = rotation / torch.linalg.norm(rotation, dim=-1, keepdim=True)
        out_device = comfy.model_management.intermediate_device()
        return (
            xyz.to(out_device).contiguous(), scaling.to(out_device).contiguous(),
            rotation.to(out_device).contiguous(), opacity.to(out_device).contiguous(),
            sh.to(out_device).contiguous(),
        )


def _quat_to_matrix(q):
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _matrix_to_quat(R):
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)
    s = torch.sqrt(torch.clamp(trace + 1, min=0)) * 2
    q[:, 0] = 0.25 * s
    denom = torch.where(s != 0, s, torch.ones_like(s))
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / denom
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / denom
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / denom
    m01 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (s == 0)
    s1 = torch.sqrt(torch.clamp(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=0)) * 2
    q[m01, 0] = (R[m01, 2, 1] - R[m01, 1, 2]) / s1[m01]
    q[m01, 1] = 0.25 * s1[m01]
    q[m01, 2] = (R[m01, 0, 1] + R[m01, 1, 0]) / s1[m01]
    q[m01, 3] = (R[m01, 0, 2] + R[m01, 2, 0]) / s1[m01]
    m11 = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (s == 0)
    s2 = torch.sqrt(torch.clamp(1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=0)) * 2
    q[m11, 0] = (R[m11, 0, 2] - R[m11, 2, 0]) / s2[m11]
    q[m11, 1] = (R[m11, 0, 1] + R[m11, 1, 0]) / s2[m11]
    q[m11, 2] = 0.25 * s2[m11]
    q[m11, 3] = (R[m11, 1, 2] + R[m11, 2, 1]) / s2[m11]
    m21 = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1]) & (s == 0)
    s3 = torch.sqrt(torch.clamp(1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=0)) * 2
    q[m21, 0] = (R[m21, 1, 0] - R[m21, 0, 1]) / s3[m21]
    q[m21, 1] = (R[m21, 0, 2] + R[m21, 2, 0]) / s3[m21]
    q[m21, 2] = (R[m21, 1, 2] + R[m21, 2, 1]) / s3[m21]
    q[m21, 3] = 0.25 * s3[m21]
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def build_gaussian_models(decoder, points_pred: dict, pred: dict):
    # Assemble GaussianModels from the elastic decoder layout. decoder is the ElasticGaussianFixedlenDecoder
    # (carries layout / rep_config / _get_offset)
    x = points_pred
    offset = decoder._get_offset(pred['features'])
    h = pred["features"]
    ret = []
    for i in range(h.shape[0]):
        g = GaussianModel(
            sh_degree=0,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            mininum_kernel_size=decoder.rep_config['filter_kernel_size_3d'],
            scaling_bias=decoder.rep_config['scaling_bias'],
            opacity_bias=decoder.rep_config['opacity_bias'],
            scaling_activation=decoder.rep_config['scaling_activation'],
            device=h.device,
        )
        _x = x["points"][i, :, None, :]
        for k, v in decoder.layout.items():
            if k == '_xyz':
                setattr(g, k, (offset[i] + _x).flatten(0, 1))
            elif k in ('_xyz_center', '_offset_scale'):
                continue
            else:
                feats = h[i][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                setattr(g, k, feats * decoder.rep_config['lr'][k])
        ret.append(g)
    return ret
