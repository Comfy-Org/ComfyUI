### Paint UNet parity: native port vs tencent/Hunyuan3D-2.1 hunyuan3d-paintpbr-v2-1

| tensor | shape | max abs diff | mean abs diff | max rel diff |
|--------|-------|--------------|---------------|--------------|
| `act/unet.conv_in` | (12, 320, 32, 32) | 1.703e-03 | 5.670e-05 | 3.707e-04 |
| `act/unet.conv_out` | (12, 4, 32, 32) | 4.888e-04 | 8.416e-05 | 3.258e-04 |
| `act/unet.down_blocks.0` | (12, 320, 16, 16) | 3.904e-03 | 2.846e-04 | 2.261e-04 |
| `act/unet.down_blocks.1` | (12, 640, 8, 8) | 7.776e-03 | 4.542e-04 | 2.985e-04 |
| `act/unet.down_blocks.2` | (12, 1280, 4, 4) | 7.889e-03 | 6.805e-04 | 2.364e-04 |
| `act/unet.down_blocks.3` | (12, 1280, 4, 4) | 7.822e-03 | 6.488e-04 | 2.491e-04 |
| `act/unet.mid_block` | (12, 1280, 4, 4) | 1.448e-02 | 6.743e-04 | 4.231e-04 |
| `act/unet.up_blocks.0` | (12, 1280, 8, 8) | 1.559e-02 | 3.453e-04 | 3.255e-04 |
| `act/unet.up_blocks.1` | (12, 1280, 16, 16) | 3.104e-02 | 1.210e-03 | 3.757e-04 |
| `act/unet.up_blocks.2` | (12, 640, 32, 32) | 3.072e-02 | 5.120e-04 | 2.502e-04 |
| `act/unet.up_blocks.3` | (12, 320, 32, 32) | 1.564e-02 | 4.431e-04 | 3.539e-04 |
| `output/noise_pred` | (12, 4, 32, 32) | 3.278e-06 | 7.093e-07 | 2.185e-06 |

bundle: `reference_v6_h32.safetensors`  input_args: `{"seed": 7, "views": 6, "height": 32, "timestep": 999}`  reference torch: `2.13.0+cpu`  native torch: `2.11.0+cu128`
