import torch
import numpy as np
from scipy.ndimage import gaussian_filter

class HeatmapHead(torch.nn.Module):
    def __init__(
            self,
            in_channels=640,
            out_channels=133,
            input_size=(768, 1024),
            heatmap_scale=4,
            deconv_out_channels=(640,),
            deconv_kernel_sizes=(4,),
            conv_out_channels=(640,),
            conv_kernel_sizes=(1,),
            final_layer_kernel_size=1,
            device=None, dtype=None, operations=None
        ):
        super().__init__()

        self.heatmap_size = (input_size[0] // heatmap_scale, input_size[1] // heatmap_scale)
        self.scale_factor = ((np.array(input_size) - 1) / (np.array(self.heatmap_size) - 1)).astype(np.float32)

        # Deconv layers
        if deconv_out_channels:
            deconv_layers = []
            for out_ch, kernel_size in zip(deconv_out_channels, deconv_kernel_sizes):
                if kernel_size == 4:
                    padding, output_padding = 1, 0
                elif kernel_size == 3:
                    padding, output_padding = 1, 1
                elif kernel_size == 2:
                    padding, output_padding = 0, 0
                else:
                    raise ValueError(f'Unsupported kernel size {kernel_size}')

                deconv_layers.extend([
                    operations.ConvTranspose2d(in_channels, out_ch, kernel_size,
                                     stride=2, padding=padding, output_padding=output_padding, bias=False, device=device, dtype=dtype),
                    torch.nn.InstanceNorm2d(out_ch, device=device, dtype=dtype),
                    torch.nn.SiLU(inplace=True)
                ])
                in_channels = out_ch
            self.deconv_layers = torch.nn.Sequential(*deconv_layers)
        else:
            self.deconv_layers = torch.nn.Identity()

        # Conv layers
        if conv_out_channels:
            conv_layers = []
            for out_ch, kernel_size in zip(conv_out_channels, conv_kernel_sizes):
                padding = (kernel_size - 1) // 2
                conv_layers.extend([
                    operations.Conv2d(in_channels, out_ch, kernel_size,
                            stride=1, padding=padding, device=device, dtype=dtype),
                    torch.nn.InstanceNorm2d(out_ch, device=device, dtype=dtype),
                    torch.nn.SiLU(inplace=True)
                ])
                in_channels = out_ch
            self.conv_layers = torch.nn.Sequential(*conv_layers)
        else:
            self.conv_layers = torch.nn.Identity()

        self.final_layer = operations.Conv2d(in_channels, out_channels, kernel_size=final_layer_kernel_size, padding=final_layer_kernel_size // 2, device=device, dtype=dtype)

    def forward(self, x): # Decode heatmaps to keypoints
        heatmaps = self.final_layer(self.conv_layers(self.deconv_layers(x)))
        heatmaps_np = heatmaps.float().cpu().numpy()  # (B, K, H, W)
        B, K, H, W = heatmaps_np.shape

        batch_keypoints = []
        batch_scores = []

        for b in range(B):
            hm = heatmaps_np[b].copy()  # (K, H, W)

            # --- vectorised argmax ---
            flat = hm.reshape(K, -1)
            idx = np.argmax(flat, axis=1)
            scores = flat[np.arange(K), idx].copy()
            y_locs, x_locs = np.unravel_index(idx, (H, W))
            keypoints = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)  # (K, 2) in heatmap space
            invalid = scores <= 0.
            keypoints[invalid] = -1

            # --- DARK sub-pixel refinement (UDP) ---
            # 1. Gaussian blur with max-preserving normalisation
            border = 5  # (kernel-1)//2 for kernel=11
            for k in range(K):
                origin_max = np.max(hm[k])
                dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
                dr[border:-border, border:-border] = hm[k].copy()
                dr = gaussian_filter(dr, sigma=2.0, truncate=2.5)
                hm[k] = dr[border:-border, border:-border].copy()
                cur_max = np.max(hm[k])
                if cur_max > 0:
                    hm[k] *= origin_max / cur_max
            # 2. Log-space for Taylor expansion
            np.clip(hm, 1e-3, 50., hm)
            np.log(hm, hm)
            # 3. Hessian-based Newton step
            hm_pad = np.pad(hm, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()
            index = keypoints[:, 0] + 1 + (keypoints[:, 1] + 1) * (W + 2)
            index += (W + 2) * (H + 2) * np.arange(0, K)
            index = index.astype(int).reshape(-1, 1)
            i_       = hm_pad[index]
            ix1      = hm_pad[index + 1]
            iy1      = hm_pad[index + W + 2]
            ix1y1    = hm_pad[index + W + 3]
            ix1_y1_  = hm_pad[index - W - 3]
            ix1_     = hm_pad[index - 1]
            iy1_     = hm_pad[index - 2 - W]
            dx = 0.5 * (ix1 - ix1_)
            dy = 0.5 * (iy1 - iy1_)
            derivative = np.concatenate([dx, dy], axis=1).reshape(K, 2, 1)
            dxx = ix1  - 2 * i_ + ix1_
            dyy = iy1  - 2 * i_ + iy1_
            dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
            hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(K, 2, 2)
            hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
            keypoints -= np.einsum('imn,ink->imk', hessian, derivative).squeeze(axis=-1)

            # --- restore to input image space ---
            keypoints = keypoints * self.scale_factor
            keypoints[invalid] = -1

            batch_keypoints.append(keypoints)
            batch_scores.append(scores)

        return batch_keypoints, batch_scores
