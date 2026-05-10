"""Three-modality dataset: graph + text + (rendered ball-stick) image."""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from pathlib import Path

import torch
from PIL import Image

from src.data.multitask_fusion_dataset import MultiTaskFusionDataset, collate_multitask_fusion


class ThreeModalityFusionDataset(MultiTaskFusionDataset):
    """MultiTaskFusionDataset that additionally returns the ResNet-preprocessed
    image tensor for each material.

    Image source: data/processed/multimodal_v1/images/<material_id>.png
    (produced once by scripts/render_cifs_to_png.py).
    """

    def __init__(
        self,
        df,
        image_dir: str = "data/processed/multimodal_v1/images",
        image_preprocess=None,
        project_root: Path = None,
        **kwargs,
    ):
        super().__init__(df, **kwargs)
        self.image_dir = Path(image_dir)
        if project_root is not None and not self.image_dir.is_absolute():
            self.image_dir = project_root / self.image_dir
        self.image_preprocess = image_preprocess  # torchvision preprocess transform
        self._image_cache = {}

    def __getitem__(self, idx: int):
        # Reuse parent for first 7 items
        atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask, target_t, is_metal_t = \
            super().__getitem__(idx)

        material_id = self.df.iloc[idx]["material_id"]
        if material_id in self._image_cache:
            image_t = self._image_cache[material_id]
        else:
            img_path = self.image_dir / f"{material_id}.png"
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                image_t = self.image_preprocess(im) if self.image_preprocess else self._default_to_tensor(im)
            self._image_cache[material_id] = image_t

        return (atom_fea, nbr_fea, nbr_idx, input_ids, attention_mask,
                target_t, is_metal_t, image_t)

    @staticmethod
    def _default_to_tensor(im):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(im)


def collate_three_modality(batch):
    """Like collate_multitask_fusion but threads the image tensor through as the 9th element."""
    base_batch = [(af, nf, ni, ii, am, t, im) for (af, nf, ni, ii, am, t, im, _) in batch]
    images = [item[7] for item in batch]

    (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
     input_ids, attention_mask, targets, is_metal) = collate_multitask_fusion(base_batch)
    images_t = torch.stack(images, dim=0)

    return (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx,
            input_ids, attention_mask, targets, is_metal, images_t)
