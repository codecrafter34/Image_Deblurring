"""
Paired Deblurring Dataset and DataLoader creation utilities.
Supports datasets such as Kaggle Real-World Image Deblurring, GoPro, RealBlur, HIDE, etc.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from .augmentations import SynchronizedAugmentation, TestTransform


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMG_EXTENSIONS


def load_image_as_tensor(path: str) -> torch.Tensor:
    """Loads an image as a float32 PyTorch tensor in range [0, 1] with shape (C, H, W)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        tensor = TF.to_tensor(img)
    return tensor


class PairedDeblurDataset(Dataset):
    """
    Dataset for paired blurry and sharp RGB images.
    
    Can be initialized either by passing explicit `blur_dir` and `sharp_dir` paths,
    or by specifying `root_dir` with a `split` (e.g., 'train', 'val', or 'test').
    In the latter case, it looks for subdirectories:
       root_dir/split/blur and root_dir/split/sharp
       OR root_dir/split/input and root_dir/split/target.
    """
    def __init__(
        self,
        blur_dir: Optional[str] = None,
        sharp_dir: Optional[str] = None,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.transform = transform
        self.pairs: List[Tuple[str, str]] = []

        if blur_dir and sharp_dir:
            self._discover_pairs_from_dirs(Path(blur_dir), Path(sharp_dir))
        elif root_dir:
            self._discover_pairs_from_root(Path(root_dir), split)
        else:
            raise ValueError("Either (blur_dir, sharp_dir) or root_dir must be provided.")

        if max_samples and max_samples < len(self.pairs):
            self.pairs = self.pairs[:max_samples]

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired images found! Check directories provided: "
                f"blur_dir={blur_dir}, sharp_dir={sharp_dir}, root_dir={root_dir}, split={split}."
            )

    def _discover_pairs_from_dirs(self, blur_dir: Path, sharp_dir: Path):
        blur_files = {p.name: p for p in blur_dir.iterdir() if is_image_file(p.name)}
        sharp_files = {p.name: p for p in sharp_dir.iterdir() if is_image_file(p.name)}

        # Find matching filenames
        common_names = sorted(list(set(blur_files.keys()).intersection(sharp_files.keys())))
        if not common_names:
            # Fallback: check matching stem if extensions differ
            blur_stems = {p.stem: p for p in blur_files.values()}
            sharp_stems = {p.stem: p for p in sharp_files.values()}
            common_stems = sorted(list(set(blur_stems.keys()).intersection(sharp_stems.keys())))
            for stem in common_stems:
                self.pairs.append((str(blur_stems[stem]), str(sharp_stems[stem])))
        else:
            for name in common_names:
                self.pairs.append((str(blur_files[name]), str(sharp_files[name])))

    def _discover_pairs_from_root(self, root_dir: Path, split: str):
        split_dir = root_dir / split
        candidate_pairs = [
            (split_dir / "blur", split_dir / "sharp"),
            (split_dir / "input", split_dir / "target"),
            (split_dir / "blurry", split_dir / "clean"),
            (root_dir / f"{split}_blur", root_dir / f"{split}_sharp"),
        ]

        found = False
        for b_dir, s_dir in candidate_pairs:
            if b_dir.is_dir() and s_dir.is_dir():
                self._discover_pairs_from_dirs(b_dir, s_dir)
                found = True
                break

        if not found:
            # Recursive search for subfolders containing 'blur' and 'sharp' (e.g. GoPro style)
            all_blur = sorted([p for p in split_dir.rglob("*") if p.parent.name == "blur" and is_image_file(p.name)])
            for b_path in all_blur:
                s_path = b_path.parent.parent / "sharp" / b_path.name
                if s_path.is_file():
                    self.pairs.append((str(b_path), str(s_path)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        blur_path, sharp_path = self.pairs[idx]
        blur_tensor = load_image_as_tensor(blur_path)
        sharp_tensor = load_image_as_tensor(sharp_path)

        if self.transform is not None:
            blur_tensor, sharp_tensor = self.transform(blur_tensor, sharp_tensor)

        filename = Path(blur_path).name
        return blur_tensor, sharp_tensor, filename


class SyntheticPairedDataset(Dataset):
    """
    Synthetic paired dataset generator for testing, debugging, or environments
    where external real-world datasets have not yet been downloaded.
    Generates synthetic sharp geometric/texture patterns and applies motion/defocus blur.
    """
    def __init__(self, num_samples: int = 64, image_size: int = 256, transform: Optional[Callable] = None):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        torch.manual_seed(idx)
        # Create structured synthetic sharp image with textures and edges
        x = torch.linspace(-3, 3, self.image_size)
        y = torch.linspace(-3, 3, self.image_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        
        c1 = torch.sin(grid_x * 5) * torch.cos(grid_y * 5)
        c2 = torch.exp(-(grid_x**2 + grid_y**2) / 2)
        c3 = (torch.sin(grid_x * 12) > 0).float()
        sharp = torch.stack([c1, c2, c3], dim=0)
        sharp = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)

        # Apply synthetic 1D/2D motion blur kernel
        kernel_size = 15
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        kernel[0, 0, kernel_size // 2, :] = 1.0 / kernel_size  # horizontal motion blur
        blur = F.conv2d(sharp.unsqueeze(0), kernel.repeat(3, 1, 1, 1), padding=kernel_size // 2, groups=3).squeeze(0)

        if self.transform is not None:
            blur, sharp = self.transform(blur, sharp)

        return blur, sharp, f"synthetic_{idx:04d}.png"


def create_dataloaders(
    root_dir: Optional[str] = None,
    train_blur_dir: Optional[str] = None,
    train_sharp_dir: Optional[str] = None,
    val_blur_dir: Optional[str] = None,
    val_sharp_dir: Optional[str] = None,
    batch_size: int = 16,
    patch_size: int = 256,
    num_workers: int = 4,
    cutblur_prob: float = 0.5,
    use_synthetic_fallback: bool = True,
    synthetic_samples: int = 64,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates PyTorch DataLoaders for train and validation splits.
    If no directory exists and use_synthetic_fallback=True, a synthetic dataset is returned.
    """
    train_transform = SynchronizedAugmentation(
        patch_size=patch_size,
        flip_prob=0.5,
        rotation=True,
        cutblur_prob=cutblur_prob,
    )
    val_transform = TestTransform(crop_size=patch_size)

    # Train dataset
    train_dataset = None
    try:
        if train_blur_dir and train_sharp_dir:
            train_dataset = PairedDeblurDataset(blur_dir=train_blur_dir, sharp_dir=train_sharp_dir, transform=train_transform)
        elif root_dir and os.path.exists(root_dir):
            train_dataset = PairedDeblurDataset(root_dir=root_dir, split="train", transform=train_transform)
    except Exception as e:
        if not use_synthetic_fallback:
            raise e

    if train_dataset is None:
        if use_synthetic_fallback:
            print(f"[DataLoader] Using SyntheticPairedDataset (num_samples={synthetic_samples}) for training.")
            train_dataset = SyntheticPairedDataset(num_samples=synthetic_samples, image_size=patch_size, transform=train_transform)
        else:
            raise FileNotFoundError("Could not find training dataset.")

    # Validation dataset
    val_dataset = None
    try:
        if val_blur_dir and val_sharp_dir:
            val_dataset = PairedDeblurDataset(blur_dir=val_blur_dir, sharp_dir=val_sharp_dir, transform=val_transform)
        elif root_dir and os.path.exists(root_dir):
            val_dataset = PairedDeblurDataset(root_dir=root_dir, split="val", transform=val_transform)
    except Exception:
        pass

    if val_dataset is None and use_synthetic_fallback:
        print("[DataLoader] Using SyntheticPairedDataset (num_samples=16) for validation.")
        val_dataset = SyntheticPairedDataset(num_samples=16, image_size=patch_size, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return train_loader, val_loader
