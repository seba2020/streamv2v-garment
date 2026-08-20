from typing import Iterable, Optional, Union

import PIL.Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor


# Label ids for the ATR label set used by mattmdjaga/segformer_b2_clothes.
CLOTHES_LABELS = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "face": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

DEFAULT_GARMENT_LABELS = ["upper_clothes", "skirt", "pants", "dress", "belt"]


class ClothesSegmenter:
    def __init__(
        self,
        model_id: str = "mattmdjaga/segformer_b2_clothes",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device=device, dtype=dtype)
        self.model.eval()

    @torch.no_grad()
    def get_mask(
        self,
        image: PIL.Image.Image,
        latent_height: int,
        latent_width: int,
        labels: Optional[Iterable[str]] = None,
        feather_px: int = 3,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float16,
        seg_size: int = 256,
    ) -> torch.Tensor:
        """Returns a [1, 1, latent_height, latent_width] mask in [0, 1]: 1
        marks pixels belonging to the requested clothing labels, 0 marks
        everything that should stay locked to the source frame.

        `seg_size` caps the resolution the segmentation model runs at
        (mask precision beyond the latent grid is wasted anyway), which is
        the dominant cost of running this per frame.
        """
        label_ids = {CLOTHES_LABELS[name] for name in (labels or DEFAULT_GARMENT_LABELS)}

        inputs = self.processor(
            images=image, size={"height": seg_size, "width": seg_size}, return_tensors="pt"
        ).pixel_values
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        logits = self.model(pixel_values=inputs).logits  # [1, num_labels, h', w'] at seg_size/4 resolution
        pred = logits.argmax(dim=1)  # [1, h', w']

        mask = torch.zeros_like(pred, dtype=torch.float32)
        for label_id in label_ids:
            mask = torch.maximum(mask, (pred == label_id).float())
        mask = mask.unsqueeze(1)  # [1, 1, h', w']

        if feather_px > 0:
            kernel_size = feather_px * 2 + 1
            mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=feather_px)

        mask = F.interpolate(mask, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
        mask = mask.clamp(0, 1)

        target_device = device or self.device
        return mask.to(device=target_device, dtype=dtype)
