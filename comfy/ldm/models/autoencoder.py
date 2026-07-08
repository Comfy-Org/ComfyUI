import logging
import math
import torch
from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union

from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from comfy.ldm.util import get_obj_from_str, instantiate_from_config
from comfy.ldm.modules.ema import LitEma
import comfy.ops
from einops import rearrange
import comfy.model_management

class DiagonalGaussianRegularizer(torch.nn.Module):
    def __init__(self, sample: bool = False):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z, None

class EmptyRegularizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, None

class AbstractAutoencoder(torch.nn.Module):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
        **kwargs,
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logging.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logging.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logging.info(f"{context}: Restored training weights")

    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logging.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        regularizer_config: Dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        self.regularization = instantiate_from_config(
            regularizer_config
        )

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        decoder_ddconfig = kwargs.pop("decoder_ddconfig", ddconfig)
        super().__init__(
            encoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Decoder",
                "params": decoder_ddconfig,
            },
            **kwargs,
        )

        if ddconfig.get("conv3d", False):
            conv_op = comfy.ops.disable_weight_init.Conv3d
        else:
            conv_op = comfy.ops.disable_weight_init.Conv2d

        self.quant_conv = conv_op(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )

        self.post_quant_conv = conv_op(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ddconfig.get("batch_norm_latent", False):
            self.bn_eps = 1e-4
            self.bn_momentum = 0.1
            self.ps = [2, 2]
            self.bn = torch.nn.BatchNorm2d(math.prod(self.ps) * ddconfig["z_channels"],
                                           eps=self.bn_eps,
                                           momentum=self.bn_momentum,
                                           affine=False,
                                           track_running_stats=True,
                                           )
            self.bn.eval()
        else:
            self.bn = None


    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)

        if self.bn is not None:
            z = rearrange(z,
                          "... c (i pi) (j pj)  -> ... (c pi pj) i j",
                          pi=self.ps[0],
                          pj=self.ps[1],
                          )

            z = torch.nn.functional.batch_norm(z,
                                               comfy.model_management.cast_to(self.bn.running_mean, dtype=z.dtype, device=z.device),
                                               comfy.model_management.cast_to(self.bn.running_var, dtype=z.dtype, device=z.device),
                                               momentum=self.bn_momentum,
                                               eps=self.bn_eps)

        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.bn is not None:
            s = torch.sqrt(comfy.model_management.cast_to(self.bn.running_var.view(1, -1, 1, 1), dtype=z.dtype, device=z.device) + self.bn_eps)
            m = comfy.model_management.cast_to(self.bn.running_mean.view(1, -1, 1, 1), dtype=z.dtype, device=z.device)
            z = z * s + m
            z = rearrange(
                z,
                "... (c pi pj) i j -> ... c (i pi) (j pj)",
                pi=self.ps[0],
                pj=self.ps[1],
            )

        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )
