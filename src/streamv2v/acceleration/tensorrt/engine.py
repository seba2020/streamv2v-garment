from typing import *

import torch
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self._last_shapes = None

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *kvo_cache,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        kvo_cache_in_shape_dict = {f"kvo_cache_in_{i}": kvo_cache.shape for i, kvo_cache in enumerate(kvo_cache)}
        kvo_cache_out_shape_dict = {}
        for i, kvo_cache in enumerate(kvo_cache):
            shape = list(kvo_cache.shape)
            shape[1] = 1
            kvo_cache_out_shape_dict[f"kvo_cache_out_{i}"] = tuple(shape)
        
        kvo_cache_in_dict = {f"kvo_cache_in_{i}": kvo_cache for i, kvo_cache in enumerate(kvo_cache)}
        
        shape_dict = {
            "sample": latent_model_input.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
            "latent": latent_model_input.shape,
            **kvo_cache_in_shape_dict,
            **kvo_cache_out_shape_dict,
        }
        
        # Only allocate buffers if shapes changed (now optimized in allocate_buffers)
        self.engine.allocate_buffers(
            shape_dict=shape_dict,
            device=latent_model_input.device,
        )

        output = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                **kvo_cache_in_dict,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        noise_pred = output["latent"]
        if len(kvo_cache) > 0:
            kvo_cache_out = [output[f"kvo_cache_out_{i}"] for i in range(len(kvo_cache))]
        else:
            kvo_cache_out = []
        return noise_pred, kvo_cache_out

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor, **kwargs):
        shape_dict = {
            "images": images.shape,
            "latent": (
                images.shape[0],
                4,
                images.shape[2] // self.vae_scale_factor,
                images.shape[3] // self.vae_scale_factor,
            ),
        }
        self.encoder.allocate_buffers(
            shape_dict=shape_dict,
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        shape_dict = {
            "latent": latent.shape,
            "images": (
                latent.shape[0],
                3,
                latent.shape[2] * self.vae_scale_factor,
                latent.shape[3] * self.vae_scale_factor,
            ),
        }
        self.decoder.allocate_buffers(
            shape_dict=shape_dict,
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
