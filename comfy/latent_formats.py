import torch

class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_dimensions = 2
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    latent_rgb_factors_reshape = None
    taesd_decoder_name = None
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 1

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]
        self.taesd_decoder_name = "taesd_decoder"

class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3651,  0.4232,  0.4341],
                    [-0.2533, -0.0042,  0.1068],
                    [ 0.1076,  0.1111, -0.0362],
                    [-0.3165, -0.2492, -0.2188]
                ]
        self.latent_rgb_factors_bias = [ 0.1084, -0.0175, -0.0011]

        self.taesd_decoder_name = "taesdxl_decoder"

class SDXL_Playground_2_5(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(1, 4, 1, 1)
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(1, 4, 1, 1)

        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3920,  0.4054,  0.4549],
                    [-0.2634, -0.0196,  0.0653],
                    [ 0.0568,  0.1687, -0.0755],
                    [-0.3112, -0.2359, -0.2076]
                ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.latent_rgb_factors = [
            [-0.2340, -0.3863, -0.3257],
            [ 0.0994,  0.0885, -0.0908],
            [-0.2833, -0.2349, -0.3741],
            [ 0.2523, -0.0055, -0.1651]
        ]

class SC_Prior(LatentFormat):
    latent_channels = 16
    spacial_downscale_ratio = 42
    def __init__(self):
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.0326, -0.0204, -0.0127],
            [-0.1592, -0.0427,  0.0216],
            [ 0.0873,  0.0638, -0.0020],
            [-0.0602,  0.0442,  0.1304],
            [ 0.0800, -0.0313, -0.1796],
            [-0.0810, -0.0638, -0.1581],
            [ 0.1791,  0.1180,  0.0967],
            [ 0.0740,  0.1416,  0.0432],
            [-0.1745, -0.1888, -0.1373],
            [ 0.2412,  0.1577,  0.0928],
            [ 0.1908,  0.0998,  0.0682],
            [ 0.0209,  0.0365, -0.0092],
            [ 0.0448, -0.0650, -0.1728],
            [-0.1658, -0.1045, -0.1308],
            [ 0.0542,  0.1545,  0.1325],
            [-0.0352, -0.1672, -0.2541]
        ]

class SC_B(LatentFormat):
    spacial_downscale_ratio = 4
    def __init__(self):
        self.scale_factor = 1.0 / 0.43
        self.latent_rgb_factors = [
            [ 0.1121,  0.2006,  0.1023],
            [-0.2093, -0.0222, -0.0195],
            [-0.3087, -0.1535,  0.0366],
            [ 0.0290, -0.1574, -0.4078]
        ]

class SD3(LatentFormat):
    latent_channels = 16
    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0922, -0.0175,  0.0749],
            [ 0.0311,  0.0633,  0.0954],
            [ 0.1994,  0.0927,  0.0458],
            [ 0.0856,  0.0339,  0.0902],
            [ 0.0587,  0.0272, -0.0496],
            [-0.0006,  0.1104,  0.0309],
            [ 0.0978,  0.0306,  0.0427],
            [-0.0042,  0.1038,  0.1358],
            [-0.0194,  0.0020,  0.0669],
            [-0.0488,  0.0130, -0.0268],
            [ 0.0922,  0.0988,  0.0951],
            [-0.0278,  0.0524, -0.0542],
            [ 0.0332,  0.0456,  0.0895],
            [-0.0069, -0.0030, -0.0810],
            [-0.0596, -0.0465, -0.0293],
            [-0.1448, -0.1463, -0.1189]
        ]
        self.latent_rgb_factors_bias = [0.2394, 0.2135, 0.1925]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class StableAudio1(LatentFormat):
    latent_channels = 64
    latent_dimensions = 1
    temporal_downscale_ratio = 2048

class StableAudio3(LatentFormat):
    latent_channels = 256
    latent_dimensions = 1
    temporal_downscale_ratio = 4096

class Flux(SD3):
    latent_channels = 16
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778]
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class Flux2(LatentFormat):
    latent_channels = 128
    spacial_downscale_ratio = 16

    def __init__(self):
        self.latent_rgb_factors =[
            [0.0058, 0.0113, 0.0073],
            [0.0495, 0.0443, 0.0836],
            [-0.0099, 0.0096, 0.0644],
            [0.2144, 0.3009, 0.3652],
            [0.0166, -0.0039, -0.0054],
            [0.0157, 0.0103, -0.0160],
            [-0.0398, 0.0902, -0.0235],
            [-0.0052, 0.0095, 0.0109],
            [-0.3527, -0.2712, -0.1666],
            [-0.0301, -0.0356, -0.0180],
            [-0.0107, 0.0078, 0.0013],
            [0.0746, 0.0090, -0.0941],
            [0.0156, 0.0169, 0.0070],
            [-0.0034, -0.0040, -0.0114],
            [0.0032, 0.0181, 0.0080],
            [-0.0939, -0.0008, 0.0186],
            [0.0018, 0.0043, 0.0104],
            [0.0284, 0.0056, -0.0127],
            [-0.0024, -0.0022, -0.0030],
            [0.1207, -0.0026, 0.0065],
            [0.0128, 0.0101, 0.0142],
            [0.0137, -0.0072, -0.0007],
            [0.0095, 0.0092, -0.0059],
            [0.0000, -0.0077, -0.0049],
            [-0.0465, -0.0204, -0.0312],
            [0.0095, 0.0012, -0.0066],
            [0.0290, -0.0034, 0.0025],
            [0.0220, 0.0169, -0.0048],
            [-0.0332, -0.0457, -0.0468],
            [-0.0085, 0.0389, 0.0609],
            [-0.0076, 0.0003, -0.0043],
            [-0.0111, -0.0460, -0.0614],
        ]

        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.latent_rgb_factors_reshape = lambda t: t.reshape(t.shape[0], 32, 2, 2, t.shape[-2], t.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(t.shape[0], 32, t.shape[-2] * 2, t.shape[-1] * 2)
        self.taesd_decoder_name = "taef2_decoder"

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent

class TripoSplat(LatentFormat):
    # Sequence latent (B, 8192, 16) the camera token rides alongside as a second nested latent
    latent_channels = 16

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent

class Mochi(LatentFormat):
    latent_channels = 12
    latent_dimensions = 3
    temporal_downscale_ratio = 6

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([-0.06730895953510081, -0.038011381506090416, -0.07477820912866141,
                                          -0.05565264470995561, 0.012767231469026969, -0.04703542746246419,
                                          0.043896967884726704, -0.09346305707025976, -0.09918314763016893,
                                          -0.008729793427399178, -0.011931556316503654, -0.0321993391887285]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([0.9263795028493863, 0.9248894543193766, 0.9393059390890617,
                                         0.959253732819592, 0.8244560132752793, 0.917259975397747,
                                         0.9294154431013696, 1.3720942357788521, 0.881393668867029,
                                         0.9168315692124348, 0.9185249279345552, 0.9274757570805041]).view(1, self.latent_channels, 1, 1, 1)

        self.latent_rgb_factors =[
            [-0.0069, -0.0045,  0.0018],
            [ 0.0154, -0.0692, -0.0274],
            [ 0.0333,  0.0019,  0.0206],
            [-0.1390,  0.0628,  0.1678],
            [-0.0725,  0.0134, -0.1898],
            [ 0.0074, -0.0270, -0.0209],
            [-0.0176, -0.0277, -0.0221],
            [ 0.5294,  0.5204,  0.3852],
            [-0.0326, -0.0446, -0.0143],
            [-0.0659,  0.0153, -0.0153],
            [ 0.0185, -0.0217,  0.0014],
            [-0.0396, -0.0495, -0.0281]
        ]
        self.latent_rgb_factors_bias = [-0.0940, -0.1418, -0.1453]
        self.taesd_decoder_name = None #TODO

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

class LTXV(LatentFormat):
    latent_channels = 128
    latent_dimensions = 3
    spacial_downscale_ratio = 32
    temporal_downscale_ratio = 8

    def __init__(self):
        self.latent_rgb_factors = [
            [ 1.1202e-02, -6.3815e-04, -1.0021e-02],
            [ 8.6031e-02,  6.5813e-02,  9.5409e-04],
            [-1.2576e-02, -7.5734e-03, -4.0528e-03],
            [ 9.4063e-03, -2.1688e-03,  2.6093e-03],
            [ 3.7636e-03,  1.2765e-02,  9.1548e-03],
            [ 2.1024e-02, -5.2973e-03,  3.4373e-03],
            [-8.8896e-03, -1.9703e-02, -1.8761e-02],
            [-1.3160e-02, -1.0523e-02,  1.9709e-03],
            [-1.5152e-03, -6.9891e-03, -7.5810e-03],
            [-1.7247e-03,  4.6560e-04, -3.3839e-03],
            [ 1.3617e-02,  4.7077e-03, -2.0045e-03],
            [ 1.0256e-02,  7.7318e-03,  1.3948e-02],
            [-1.6108e-02, -6.2151e-03,  1.1561e-03],
            [ 7.3407e-03,  1.5628e-02,  4.4865e-04],
            [ 9.5357e-04, -2.9518e-03, -1.4760e-02],
            [ 1.9143e-02,  1.0868e-02,  1.2264e-02],
            [ 4.4575e-03,  3.6682e-05, -6.8508e-03],
            [-4.5681e-04,  3.2570e-03,  7.7929e-03],
            [ 3.3902e-02,  3.3405e-02,  3.7454e-02],
            [-2.3001e-02, -2.4877e-03, -3.1033e-03],
            [ 5.0265e-02,  3.8841e-02,  3.3539e-02],
            [-4.1018e-03, -1.1095e-03,  1.5859e-03],
            [-1.2689e-01, -1.3107e-01, -2.1005e-01],
            [ 2.6276e-02,  1.4189e-02, -3.5963e-03],
            [-4.8679e-03,  8.8486e-03,  7.8029e-03],
            [-1.6610e-03, -4.8597e-03, -5.2060e-03],
            [-2.1010e-03,  2.3610e-03,  9.3796e-03],
            [-2.2482e-02, -2.1305e-02, -1.5087e-02],
            [-1.5753e-02, -1.0646e-02, -6.5083e-03],
            [-4.6975e-03,  5.0288e-03, -6.7390e-03],
            [ 1.1951e-02,  2.0712e-02,  1.6191e-02],
            [-6.3704e-03, -8.4827e-03, -9.5483e-03],
            [ 7.2610e-03, -9.9326e-03, -2.2978e-02],
            [-9.1904e-04,  6.2882e-03,  9.5720e-03],
            [-3.7178e-02, -3.7123e-02, -5.6713e-02],
            [-1.3373e-01, -1.0720e-01, -5.3801e-02],
            [-5.3702e-03,  8.1256e-03,  8.8397e-03],
            [-1.5247e-01, -2.1437e-01, -2.1843e-01],
            [ 3.1441e-02,  7.0335e-03, -9.7541e-03],
            [ 2.1528e-03, -8.9817e-03, -2.1023e-02],
            [ 3.8461e-03, -5.8957e-03, -1.5014e-02],
            [-4.3470e-03, -1.2940e-02, -1.5972e-02],
            [-5.4781e-03, -1.0842e-02, -3.0204e-03],
            [-6.5347e-03,  3.0806e-03, -1.0163e-02],
            [-5.0414e-03, -7.1503e-03, -8.9686e-04],
            [-8.5851e-03, -2.4351e-03,  1.0674e-03],
            [-9.0016e-03, -9.6493e-03,  1.5692e-03],
            [ 5.0914e-03,  1.2099e-02,  1.9968e-02],
            [ 1.3758e-02,  1.1669e-02,  8.1958e-03],
            [-1.0518e-02, -1.1575e-02, -4.1307e-03],
            [-2.8410e-02, -3.1266e-02, -2.2149e-02],
            [ 2.9336e-03,  3.6511e-02,  1.8717e-02],
            [-1.6703e-02, -1.6696e-02, -4.4529e-03],
            [ 4.8818e-02,  4.0063e-02,  8.7410e-03],
            [-1.5066e-02, -5.7328e-04,  2.9785e-03],
            [-1.7613e-02, -8.1034e-03,  1.3086e-02],
            [-9.2633e-03,  1.0803e-02, -6.3489e-03],
            [ 3.0851e-03,  4.7750e-04,  1.2347e-02],
            [-2.2785e-02, -2.3043e-02, -2.6005e-02],
            [-2.4787e-02, -1.5389e-02, -2.2104e-02],
            [-2.3572e-02,  1.0544e-03,  1.2361e-02],
            [-7.8915e-03, -1.2271e-03, -6.0968e-03],
            [-1.1478e-02, -1.2543e-03,  6.2679e-03],
            [-5.4229e-02,  2.6644e-02,  6.3394e-03],
            [ 4.4216e-03, -7.3338e-03, -1.0464e-02],
            [-4.5013e-03,  1.6082e-03,  1.4420e-02],
            [ 1.3673e-02,  8.8877e-03,  4.1253e-03],
            [-1.0145e-02,  9.0072e-03,  1.5695e-02],
            [-5.6234e-03,  1.1847e-03,  8.1261e-03],
            [-3.7171e-03, -5.3538e-03,  1.2590e-03],
            [ 2.9476e-02,  2.1424e-02,  3.0424e-02],
            [-3.4925e-02, -2.4340e-02, -2.5316e-02],
            [-3.4127e-02, -2.2406e-02, -1.0589e-02],
            [-1.7342e-02, -1.3249e-02, -1.0719e-02],
            [-2.1478e-03, -8.6051e-03, -2.9878e-03],
            [ 1.2089e-03, -4.2391e-03, -6.8569e-03],
            [ 9.0411e-04, -6.6886e-03, -6.7547e-05],
            [ 1.6048e-02, -1.0057e-02, -2.8929e-02],
            [ 1.2290e-03,  1.0163e-02,  1.8861e-02],
            [ 1.7264e-02,  2.7257e-04,  1.3785e-02],
            [-1.3482e-02, -3.6427e-03,  6.7481e-04],
            [ 4.6782e-03, -5.2423e-03,  2.4467e-03],
            [-5.9113e-03, -6.2244e-03, -1.8162e-03],
            [ 1.5496e-02,  1.4582e-02,  1.9514e-03],
            [ 7.4958e-03,  1.5886e-03, -8.2305e-03],
            [ 1.9086e-02,  1.6360e-03, -3.9674e-03],
            [-5.7021e-03, -2.7307e-03, -4.1066e-03],
            [ 1.7450e-03,  1.4602e-02,  2.5794e-02],
            [-8.2788e-04,  2.2902e-03,  4.5161e-03],
            [ 1.1632e-02,  8.9193e-03, -7.2813e-03],
            [ 7.5721e-03,  2.6784e-03,  1.1393e-02],
            [ 5.1939e-03,  3.6903e-03,  1.4049e-02],
            [-1.8383e-02, -2.2529e-02, -2.4477e-02],
            [ 5.8842e-04, -5.7874e-03, -1.4770e-02],
            [-1.6125e-02, -8.6101e-03, -1.4533e-02],
            [ 2.0540e-02,  2.0729e-02,  6.4338e-03],
            [ 3.3587e-03, -1.1226e-02, -1.6444e-02],
            [-1.4742e-03, -1.0489e-02,  1.7097e-03],
            [ 2.8130e-02,  2.3546e-02,  3.2791e-02],
            [-1.8532e-02, -1.2842e-02, -8.7756e-03],
            [-8.0533e-03, -1.0771e-02, -1.7536e-02],
            [-3.9009e-03,  1.6150e-02,  3.3359e-02],
            [-7.4554e-03, -1.4154e-02, -6.1910e-03],
            [ 3.4734e-03, -1.1370e-02, -1.0581e-02],
            [ 1.1476e-02,  3.9281e-03,  2.8231e-03],
            [ 7.1639e-03, -1.4741e-03, -3.8066e-03],
            [ 2.2250e-03, -8.7552e-03, -9.5719e-03],
            [ 2.4146e-02,  2.1696e-02,  2.8056e-02],
            [-5.4365e-03, -2.4291e-02, -1.7802e-02],
            [ 7.4263e-03,  1.0510e-02,  1.2705e-02],
            [ 6.2669e-03,  6.2658e-03,  1.9211e-02],
            [ 1.6378e-02,  9.4933e-03,  6.6971e-03],
            [ 1.7173e-02,  2.3601e-02,  2.3296e-02],
            [-1.4568e-02, -9.8279e-03, -1.1556e-02],
            [ 1.4431e-02,  1.4430e-02,  6.6362e-03],
            [-6.8230e-03,  1.8863e-02,  1.4555e-02],
            [ 6.1156e-03,  3.4700e-03, -2.6662e-03],
            [-2.6983e-03, -5.9402e-03, -9.2276e-03],
            [ 1.0235e-02,  7.4173e-03, -7.6243e-03],
            [-1.3255e-02,  1.9322e-02, -9.2153e-04],
            [ 2.4222e-03, -4.8039e-03, -1.5759e-02],
            [ 2.6244e-02,  2.5951e-02,  2.0249e-02],
            [ 1.5711e-02,  1.8498e-02,  2.7407e-03],
            [-2.1714e-03,  4.7214e-03, -2.2443e-02],
            [-7.4747e-03,  7.4166e-03,  1.4430e-02],
            [-8.3906e-03, -7.9776e-03,  9.7927e-03],
            [ 3.8321e-02,  9.6622e-03, -1.9268e-02],
            [-1.4605e-02, -6.7032e-03,  3.9675e-03]
        ]

        self.latent_rgb_factors_bias = [-0.0571, -0.1657, -0.2512]

class LTXAV(LTXV):
    def __init__(self):
        self.latent_rgb_factors = None
        self.latent_rgb_factors_bias = None

class HunyuanVideo(LatentFormat):
    latent_channels = 16
    latent_dimensions = 3
    temporal_downscale_ratio = 4
    scale_factor = 0.476986
    latent_rgb_factors = [
        [-0.0395, -0.0331,  0.0445],
        [ 0.0696,  0.0795,  0.0518],
        [ 0.0135, -0.0945, -0.0282],
        [ 0.0108, -0.0250, -0.0765],
        [-0.0209,  0.0032,  0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991,  0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [ 0.1166,  0.1627,  0.0962],
        [ 0.1165,  0.0432,  0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270,  0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [ 0.0249, -0.0469, -0.1703]
    ]

    latent_rgb_factors_bias = [ 0.0259, -0.0192, -0.0761]
    taesd_decoder_name = "taehv"

class Cosmos1CV8x8x8(LatentFormat):
    latent_channels = 16
    latent_dimensions = 3
    temporal_downscale_ratio = 8

    latent_rgb_factors = [
        [ 0.1817,  0.2284,  0.2423],
        [-0.0586, -0.0862, -0.3108],
        [-0.4703, -0.4255, -0.3995],
        [ 0.0803,  0.1963,  0.1001],
        [-0.0820, -0.1050,  0.0400],
        [ 0.2511,  0.3098,  0.2787],
        [-0.1830, -0.2117, -0.0040],
        [-0.0621, -0.2187, -0.0939],
        [ 0.3619,  0.1082,  0.1455],
        [ 0.3164,  0.3922,  0.2575],
        [ 0.1152,  0.0231, -0.0462],
        [-0.1434, -0.3609, -0.3665],
        [ 0.0635,  0.1471,  0.1680],
        [-0.3635, -0.1963, -0.3248],
        [-0.1865,  0.0365,  0.2346],
        [ 0.0447,  0.0994,  0.0881]
    ]

    latent_rgb_factors_bias = [-0.1223, -0.1889, -0.1976]

class Wan21(LatentFormat):
    latent_channels = 16
    latent_dimensions = 3
    temporal_downscale_ratio = 4

    latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861]
        ]

    latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, self.latent_channels, 1, 1, 1)


        self.taesd_decoder_name = "lighttaew2_1"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean

class Wan22(Wan21):
    latent_channels = 48
    latent_dimensions = 3
    spacial_downscale_ratio = 16

    latent_rgb_factors = [
            [ 0.0119,  0.0103,  0.0046],
            [-0.1062, -0.0504,  0.0165],
            [ 0.0140,  0.0409,  0.0491],
            [-0.0813, -0.0677,  0.0607],
            [ 0.0656,  0.0851,  0.0808],
            [ 0.0264,  0.0463,  0.0912],
            [ 0.0295,  0.0326,  0.0590],
            [-0.0244, -0.0270,  0.0025],
            [ 0.0443, -0.0102,  0.0288],
            [-0.0465, -0.0090, -0.0205],
            [ 0.0359,  0.0236,  0.0082],
            [-0.0776,  0.0854,  0.1048],
            [ 0.0564,  0.0264,  0.0561],
            [ 0.0006,  0.0594,  0.0418],
            [-0.0319, -0.0542, -0.0637],
            [-0.0268,  0.0024,  0.0260],
            [ 0.0539,  0.0265,  0.0358],
            [-0.0359, -0.0312, -0.0287],
            [-0.0285, -0.1032, -0.1237],
            [ 0.1041,  0.0537,  0.0622],
            [-0.0086, -0.0374, -0.0051],
            [ 0.0390,  0.0670,  0.2863],
            [ 0.0069,  0.0144,  0.0082],
            [ 0.0006, -0.0167,  0.0079],
            [ 0.0313, -0.0574, -0.0232],
            [-0.1454, -0.0902, -0.0481],
            [ 0.0714,  0.0827,  0.0447],
            [-0.0304, -0.0574, -0.0196],
            [ 0.0401,  0.0384,  0.0204],
            [-0.0758, -0.0297, -0.0014],
            [ 0.0568,  0.1307,  0.1372],
            [-0.0055, -0.0310, -0.0380],
            [ 0.0239, -0.0305,  0.0325],
            [-0.0663, -0.0673, -0.0140],
            [-0.0416, -0.0047, -0.0023],
            [ 0.0166,  0.0112, -0.0093],
            [-0.0211,  0.0011,  0.0331],
            [ 0.1833,  0.1466,  0.2250],
            [-0.0368,  0.0370,  0.0295],
            [-0.3441, -0.3543, -0.2008],
            [-0.0479, -0.0489, -0.0420],
            [-0.0660, -0.0153,  0.0800],
            [-0.0101,  0.0068,  0.0156],
            [-0.0690, -0.0452, -0.0927],
            [-0.0145,  0.0041,  0.0015],
            [ 0.0421,  0.0451,  0.0373],
            [ 0.0504, -0.0483, -0.0356],
            [-0.0837,  0.0168,  0.0055]
        ]

    latent_rgb_factors_bias = [0.0317, -0.0878, -0.1388]

    def __init__(self):
        self.scale_factor = 1.0
        self.taesd_decoder_name = "lighttaew2_2"
        self.latents_mean = torch.tensor([
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
            ]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
            ]).view(1, self.latent_channels, 1, 1, 1)

class HunyuanImage21(LatentFormat):
    latent_channels = 64
    latent_dimensions = 2
    spacial_downscale_ratio = 32
    scale_factor = 0.75289

    latent_rgb_factors = [
        [-0.0154, -0.0397, -0.0521],
        [ 0.0005,  0.0093,  0.0006],
        [-0.0805, -0.0773, -0.0586],
        [-0.0494, -0.0487, -0.0498],
        [-0.0212, -0.0076, -0.0261],
        [-0.0179, -0.0417, -0.0505],
        [ 0.0158,  0.0310,  0.0239],
        [ 0.0409,  0.0516,  0.0201],
        [ 0.0350,  0.0553,  0.0036],
        [-0.0447, -0.0327, -0.0479],
        [-0.0038, -0.0221, -0.0365],
        [-0.0423, -0.0718, -0.0654],
        [ 0.0039,  0.0368,  0.0104],
        [ 0.0655,  0.0217,  0.0122],
        [ 0.0490,  0.1638,  0.2053],
        [ 0.0932,  0.0829,  0.0650],
        [-0.0186, -0.0209, -0.0135],
        [-0.0080, -0.0076, -0.0148],
        [-0.0284, -0.0201,  0.0011],
        [-0.0642, -0.0294, -0.0777],
        [-0.0035,  0.0076, -0.0140],
        [ 0.0519,  0.0731,  0.0887],
        [-0.0102,  0.0095,  0.0704],
        [ 0.0068,  0.0218, -0.0023],
        [-0.0726, -0.0486, -0.0519],
        [ 0.0260,  0.0295,  0.0263],
        [ 0.0250,  0.0333,  0.0341],
        [ 0.0168, -0.0120, -0.0174],
        [ 0.0226,  0.1037,  0.0114],
        [ 0.2577,  0.1906,  0.1604],
        [-0.0646, -0.0137, -0.0018],
        [-0.0112,  0.0309,  0.0358],
        [-0.0347,  0.0146, -0.0481],
        [ 0.0234,  0.0179,  0.0201],
        [ 0.0157,  0.0313,  0.0225],
        [ 0.0423,  0.0675,  0.0524],
        [-0.0031,  0.0027, -0.0255],
        [ 0.0447,  0.0555,  0.0330],
        [-0.0152,  0.0103,  0.0299],
        [-0.0755, -0.0489, -0.0635],
        [ 0.0853,  0.0788,  0.1017],
        [-0.0272, -0.0294, -0.0471],
        [ 0.0440,  0.0400, -0.0137],
        [ 0.0335,  0.0317, -0.0036],
        [-0.0344, -0.0621, -0.0984],
        [-0.0127, -0.0630, -0.0620],
        [-0.0648,  0.0360,  0.0924],
        [-0.0781, -0.0801, -0.0409],
        [ 0.0363,  0.0613,  0.0499],
        [ 0.0238,  0.0034,  0.0041],
        [-0.0135,  0.0258,  0.0310],
        [ 0.0614,  0.1086,  0.0589],
        [ 0.0428,  0.0350,  0.0205],
        [ 0.0153,  0.0173, -0.0018],
        [-0.0288, -0.0455, -0.0091],
        [ 0.0344,  0.0109, -0.0157],
        [-0.0205, -0.0247, -0.0187],
        [ 0.0487,  0.0126,  0.0064],
        [-0.0220, -0.0013,  0.0074],
        [-0.0203, -0.0094, -0.0048],
        [-0.0719,  0.0429, -0.0442],
        [ 0.1042,  0.0497,  0.0356],
        [-0.0659, -0.0578, -0.0280],
        [-0.0060, -0.0322, -0.0234]]

    latent_rgb_factors_bias = [0.0007, -0.0256, -0.0206]

class HunyuanImage21Refiner(LatentFormat):
    latent_channels = 64
    latent_dimensions = 3
    scale_factor = 1.03682

    def process_in(self, latent):
        out = latent * self.scale_factor
        out = torch.cat((out[:, :, :1], out), dim=2)
        out = out.permute(0, 2, 1, 3, 4)
        b, f_times_2, c, h, w = out.shape
        out = out.reshape(b, f_times_2 // 2, 2 * c, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out

    def process_out(self, latent):
        z = latent / self.scale_factor
        z = z.permute(0, 2, 1, 3, 4)
        b, f, c, h, w = z.shape
        z = z.reshape(b, f, 2, c // 2, h, w)
        z = z.permute(0, 1, 2, 3, 4, 5).reshape(b, f * 2, c // 2, h, w)
        z = z.permute(0, 2, 1, 3, 4)
        z = z[:, :, 1:]
        return z

class HunyuanVideo15(LatentFormat):
    latent_rgb_factors = [
        [ 0.0568, -0.0521, -0.0131],
        [ 0.0014,  0.0735,  0.0326],
        [ 0.0186,  0.0531, -0.0138],
        [-0.0031,  0.0051,  0.0288],
        [ 0.0110,  0.0556,  0.0432],
        [-0.0041, -0.0023, -0.0485],
        [ 0.0530,  0.0413,  0.0253],
        [ 0.0283,  0.0251,  0.0339],
        [ 0.0277, -0.0372, -0.0093],
        [ 0.0393,  0.0944,  0.1131],
        [ 0.0020,  0.0251,  0.0037],
        [-0.0017,  0.0012,  0.0234],
        [ 0.0468,  0.0436,  0.0203],
        [ 0.0354,  0.0439, -0.0233],
        [ 0.0090,  0.0123,  0.0346],
        [ 0.0382,  0.0029,  0.0217],
        [ 0.0261, -0.0300,  0.0030],
        [-0.0088, -0.0220, -0.0283],
        [-0.0272, -0.0121, -0.0363],
        [-0.0664, -0.0622,  0.0144],
        [ 0.0414,  0.0479,  0.0529],
        [ 0.0355,  0.0612, -0.0247],
        [ 0.0147,  0.0264,  0.0174],
        [ 0.0438,  0.0038,  0.0542],
        [ 0.0431, -0.0573, -0.0033],
        [-0.0162, -0.0211, -0.0406],
        [-0.0487, -0.0295, -0.0393],
        [ 0.0005, -0.0109,  0.0253],
        [ 0.0296,  0.0591,  0.0353],
        [ 0.0119,  0.0181, -0.0306],
        [-0.0085, -0.0362,  0.0229],
        [ 0.0005, -0.0106,  0.0242]
    ]

    latent_rgb_factors_bias = [ 0.0456, -0.0202, -0.0644]
    latent_channels = 32
    latent_dimensions = 3
    spacial_downscale_ratio = 16
    temporal_downscale_ratio = 4
    scale_factor = 1.03682
    taesd_decoder_name = "lighttaehy1_5"

class Hunyuan3Dv2(LatentFormat):
    latent_channels = 64
    latent_dimensions = 1
    scale_factor = 0.9990943042622529

class Hunyuan3Dv2_1(LatentFormat):
    scale_factor = 1.0039506158752403
    latent_channels = 64
    latent_dimensions = 1

class Hunyuan3Dv2mini(LatentFormat):
    latent_channels = 64
    latent_dimensions = 1
    scale_factor = 1.0188137142395404

class ACEAudio(LatentFormat):
    latent_channels = 8
    latent_dimensions = 2

class ACEAudio15(LatentFormat):
    latent_channels = 64
    latent_dimensions = 1
    temporal_downscale_ratio = 1764

class ChromaRadiance(LatentFormat):
    latent_channels = 3
    spacial_downscale_ratio = 1

    def __init__(self):
        self.latent_rgb_factors = [
            # R    G    B
            [ 1.0, 0.0, 0.0 ],
            [ 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 1.0 ]
        ]

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class ZImagePixelSpace(ChromaRadiance):
    """Pixel-space latent format for ZImage DCT variant.
    No VAE encoding/decoding — the model operates directly on RGB pixels.
    """
    pass

class HiDreamO1Pixel(ChromaRadiance):
    """Pixel-space latent format for HiDream-O1.
    No VAE — model patches/unpatches raw RGB internally with patch_size=32.
    """
    pass

class PixelDiTPixel(ChromaRadiance):
    pass

class CogVideoX(LatentFormat):
    """Latent format for CogVideoX-2b (THUDM/CogVideoX-2b).

    scale_factor matches the vae/config.json scaling_factor for the 2b variant.
    The 5b-class checkpoints (CogVideoX-5b, CogVideoX-1.5-5B, CogVideoX-Fun-V1.5-*)
    use a different value; see CogVideoX1_5 below.
    """
    latent_channels = 16
    latent_dimensions = 3
    temporal_downscale_ratio = 4

    def __init__(self):
        self.scale_factor = 1.15258426


class CogVideoX1_5(CogVideoX):
    """Latent format for 5b-class CogVideoX checkpoints.

    Covers THUDM/CogVideoX-5b, THUDM/CogVideoX-1.5-5B, and the CogVideoX-Fun
    V1.5-5b family (including VOID inpainting). All of these have
    scaling_factor=0.7 in their vae/config.json. Auto-selected in
    supported_models.CogVideoX_T2V based on transformer hidden dim.
    """
    def __init__(self):
        self.scale_factor = 0.7
