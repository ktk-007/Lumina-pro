import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=256, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.paths = []

        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            import glob
            self.paths += glob.glob(os.path.join(root_dir, '**', ext), recursive=True)

        print(f"Found {len(self.paths)} images in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # --- Load & force resize FIRST ---
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except:
            img = Image.new('RGB', (self.img_size, self.img_size))

        # Force exact size immediately
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        # --- Augmentation ---
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)

            # Crop then resize BACK to exact size
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(1.0, 1.0)
            )
            img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))

        # --- Convert to numpy for Lab ---
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_lab = rgb2lab(img_np).astype(np.float32)

        # --- Split L and ab ---
        L  = img_lab[:, :, 0]    # range [0, 100]
        ab = img_lab[:, :, 1:]   # range [-128, 127]

        # --- Normalize ---
        L  = L  / 100.0              # -> [0, 1]
        ab = ab / 128.0              # -> [-1, 1]

        # --- Noise augmentation on L only (Noisy Student style) ---
        if self.augment:
            # Gaussian noise
            noise = np.random.normal(0, 0.02, L.shape).astype(np.float32)
            L = np.clip(L + noise, 0, 1)

            # Salt & pepper noise
            mask = np.random.random(L.shape)
            L[mask < 0.005] = 0.0
            L[mask > 0.995] = 1.0

        # --- To tensors ---
        L_tensor  = torch.from_numpy(L).unsqueeze(0)       # (1, H, W)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)  # (2, H, W)

        return L_tensor, ab_tensor
