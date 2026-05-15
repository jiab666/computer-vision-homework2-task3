from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet


class OxfordPetSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 256,
        download: bool = True,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if split not in {"trainval", "test"}:
            raise ValueError("split must be 'trainval' or 'test'")

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        OxfordIIITPet(root=root, split=split, target_types="segmentation", download=download)

        annotation_file = self.root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
        self.samples = []
        with annotation_file.open("r", encoding="utf-8") as f:
            for line in f:
                image_id = line.strip().split(" ")[0]
                image_path = self.root / "oxford-iiit-pet" / "images" / f"{image_id}.jpg"
                mask_path = self.root / "oxford-iiit-pet" / "annotations" / "trimaps" / f"{image_id}.png"
                self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        return mask.resize((self.image_size, self.image_size), Image.NEAREST)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        mask_np = __import__("numpy").array(mask, dtype="int64")
        # Original trimap labels are {1, 2, 3}; remap to {0, 1, 2}.
        mask_np = mask_np - 1
        return torch.from_numpy(mask_np).long()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self._resize_image(image)
        mask = self._resize_mask(mask)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = self._image_to_tensor(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = self._mask_to_tensor(mask)

        return image, mask
