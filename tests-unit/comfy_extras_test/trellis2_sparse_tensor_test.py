import torch

if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

from comfy.ldm.trellis2.vae import SparseTensor


def test_sparse_tensor_spatial_shape_casts_int32_coords_to_int64(monkeypatch):
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
    feats = torch.zeros((2, 4), dtype=torch.float32)
    tensor = SparseTensor(feats=feats, coords=coords)

    seen_dtypes = []
    orig_max = torch.Tensor.max

    def spy_max(self, *args, **kwargs):
        seen_dtypes.append(self.dtype)
        return orig_max(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "max", spy_max)
    assert tensor.spatial_shape == torch.Size([5, 6, 7])
    assert seen_dtypes == [torch.int64]
