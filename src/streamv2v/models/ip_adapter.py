from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import PIL.Image
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class ImageProjModel(nn.Module):
    """Projects a pooled CLIP image embedding into `clip_extra_context_tokens`
    extra cross-attention tokens. Matches the layout of h94/IP-Adapter's
    `image_proj` checkpoint for the (non-Plus) SD1.5 adapter.
    """

    def __init__(self, cross_attention_dim: int = 768, clip_embeddings_dim: int = 1024, clip_extra_context_tokens: int = 4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(tokens)


class IPAdapterAttnWeights(nn.Module):
    """Holds the extra to_k_ip / to_v_ip projections for one UNet cross-attention layer."""

    def __init__(self, hidden_size: int, cross_attention_dim: int):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)


class _EmptySlot(nn.Module):
    """Placeholder for a self-attention position in the checkpoint's flat module list."""
    pass


class IPAdapterState:
    """Mutable, shared holder for the current image-prompt embeddings.

    Every IP-Adapter-aware attention processor reads from the same instance,
    so updating the reference garment image doesn't require reconstructing
    the UNet's attention processors or threading new arguments through the
    forked diffusers UNet.forward.
    """

    def __init__(self):
        self.hidden_states: Optional[torch.Tensor] = None  # [1, num_tokens, cross_attention_dim]

    def get(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.hidden_states is None:
            return None
        if self.hidden_states.shape[0] == batch_size:
            return self.hidden_states
        return self.hidden_states.expand(batch_size, -1, -1)


def _cross_attn_hidden_size(unet, name: str) -> int:
    if name.startswith("mid_block"):
        return unet.config.block_out_channels[-1]
    if name.startswith("up_blocks"):
        block_id = int(name.split(".")[1])
        return list(reversed(unet.config.block_out_channels))[block_id]
    if name.startswith("down_blocks"):
        block_id = int(name.split(".")[1])
        return unet.config.block_out_channels[block_id]
    raise ValueError(f"Unrecognized attention processor name: {name}")


def load_ip_adapter_sd15(
    unet,
    pretrained_model_name_or_path: str = "h94/IP-Adapter",
    subfolder: str = "models",
    weight_name: str = "ip-adapter_sd15.bin",
    image_encoder_subfolder: str = "models/image_encoder",
    num_tokens: int = 4,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Loads h94/IP-Adapter's SD1.5 checkpoint and returns the pieces needed
    to encode a reference image and to condition the UNet's cross-attention
    layers, without depending on diffusers' own (version-specific) IP-Adapter
    loading machinery.

    Returns
    -------
    image_encoder, image_processor, image_proj_model, ip_layers_by_name
        `ip_layers_by_name` maps each cross-attention processor name (as in
        `unet.attn_processors`) to its `IPAdapterAttnWeights`.
    """
    ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=weight_name)
    state_dict = torch.load(ckpt_path, map_location="cpu")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder=image_encoder_subfolder
    ).to(device=device, dtype=dtype)
    image_encoder.eval()
    image_processor = CLIPImageProcessor()

    cross_attention_dim = unet.config.cross_attention_dim
    clip_embeddings_dim = image_encoder.config.projection_dim

    image_proj_model = ImageProjModel(
        cross_attention_dim=cross_attention_dim,
        clip_embeddings_dim=clip_embeddings_dim,
        clip_extra_context_tokens=num_tokens,
    )
    image_proj_model.load_state_dict(state_dict["image_proj"])
    image_proj_model = image_proj_model.to(device=device, dtype=dtype)
    image_proj_model.eval()

    # h94/IP-Adapter's "ip_adapter" state dict was originally saved from
    # nn.ModuleList(unet.attn_processors.values()).state_dict(), i.e. its
    # keys are indexed by position across ALL attention processors
    # (self-attention slots contribute no keys). Rebuilding that exact
    # module list and loading into it sidesteps any guesswork about the
    # numbering scheme.
    cross_attn_names = [name for name in unet.attn_processors.keys() if name.endswith("attn2.processor")]
    ip_layers_by_name: Dict[str, IPAdapterAttnWeights] = {}
    modules_in_order = []
    for name in unet.attn_processors.keys():
        if name in cross_attn_names:
            layer = IPAdapterAttnWeights(
                hidden_size=_cross_attn_hidden_size(unet, name),
                cross_attention_dim=cross_attention_dim,
            )
            ip_layers_by_name[name] = layer
            modules_in_order.append(layer)
        else:
            modules_in_order.append(_EmptySlot())

    ip_layers = nn.ModuleList(modules_in_order)
    ip_layers.load_state_dict(state_dict["ip_adapter"])
    for layer in ip_layers_by_name.values():
        layer.to(device=device, dtype=dtype)
        layer.eval()

    return image_encoder, image_processor, image_proj_model, ip_layers_by_name


class IPAdapterEncoder:
    """Encodes a reference PIL image into IP-Adapter image-prompt tokens."""

    def __init__(self, image_encoder, image_processor, image_proj_model, device, dtype):
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.image_proj_model = image_proj_model
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, image: PIL.Image.Image) -> torch.Tensor:
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(pixel_values).image_embeds
        return self.image_proj_model(clip_image_embeds)
