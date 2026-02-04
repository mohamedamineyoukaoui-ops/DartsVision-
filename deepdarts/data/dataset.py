"""
DeepDarts PyTorch Dataloader for Real Dataset
Loads the official DeepDarts dataset from IEEE Dataport.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

# Dataset splits (from official repo)
D1_VAL = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
D1_TEST = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 
           'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']

D2_VAL = ['d2_02_03_2021', 'd2_02_05_2021']
D2_TEST = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']


class DeepDartsDataset(Dataset):
    """
    Official DeepDarts dataset loader.
    
    Directory structure expected:
        dataset/
            labels.pkl
            cropped_images/
                800/
                    d1_02_04_2020/
                        IMG_1081.JPG
                        ...
    """
    
    MAX_KEYPOINTS = 7  # 4 calibration + up to 3 darts
    NUM_CLASSES = 5    # 0=dart, 1-4=calibration points
    
    def __init__(
        self,
        labels_path: str,
        images_path: str,
        dataset: str = 'd1',
        split: str = 'train',
        input_size: int = 800,
        bbox_size: float = 0.025,
        transform=None
    ):
        """
        Args:
            labels_path: Path to labels.pkl
            images_path: Path to cropped_images/{size}/ directory
            dataset: 'd1' or 'd2'
            split: 'train', 'val', or 'test'
            input_size: Image size (should match cropped images)
            bbox_size: Bounding box size as fraction of input
            transform: Optional augmentation transforms
        """
        self.images_path = Path(images_path)
        self.input_size = input_size
        self.bbox_size = bbox_size
        self.transform = transform
        
        # Load and filter labels
        df = pd.read_pickle(labels_path)
        df = df[df.img_folder.str.contains(dataset)]
        
        # Get splits
        if dataset == 'd1':
            val_folders, test_folders = D1_VAL, D1_TEST
        else:
            val_folders, test_folders = D2_VAL, D2_TEST
        
        if split == 'train':
            self.data = df[~np.isin(df.img_folder, val_folders + test_folders)]
        elif split == 'val':
            self.data = df[np.isin(df.img_folder, val_folders)]
        else:  # test
            self.data = df[np.isin(df.img_folder, test_folders)]
        
        self.data = self.data.reset_index(drop=True)
        print(f"Loaded {len(self.data)} images for {dataset}/{split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.images_path / row['img_folder'] / row['img_name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            # Return blank if image doesn't exist
            print(f"Warning: Could not load {img_path}")
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get keypoints (xy has variable length, up to 7 points with x, y)
        xy_raw = np.array(row['xy'])  # Shape: (N, 2) where N <= 7
        
        # Create full keypoint array with visibility
        xy = np.zeros((self.MAX_KEYPOINTS, 3), dtype=np.float32)
        n_points = min(xy_raw.shape[0], self.MAX_KEYPOINTS)
        xy[:n_points, :2] = xy_raw[:n_points]
        xy[:n_points, 2] = 1.0  # visibility
        
        # Apply augmentation
        if self.transform is not None:
            image, xy = self.transform(image, xy)
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Create bounding box targets
        # Format: (x, y, w, h, class) for each keypoint
        bboxes = self._create_bboxes(xy)
        
        return image, bboxes, xy
    
    def _create_bboxes(self, xy):
        """Create bounding box targets from keypoints."""
        bboxes = []
        
        for i, pt in enumerate(xy):
            if pt[2] == 0:  # not visible
                continue
            
            x, y = pt[0], pt[1]
            
            # Skip if too close to edge
            half = self.bbox_size / 2
            if x - half <= 0 or x + half >= 1 or y - half <= 0 or y + half >= 1:
                continue
            
            # Class: 0 for darts (indices 4-6), 1-4 for calibration (indices 0-3)
            if i < 4:
                cls = i + 1  # Calibration points are classes 1-4
            else:
                cls = 0  # Darts are class 0
            
            bboxes.append([x, y, self.bbox_size, self.bbox_size, cls])
        
        if len(bboxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        return torch.tensor(bboxes, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate for variable-length bboxes."""
    images = torch.stack([item[0] for item in batch])
    bboxes = [item[1] for item in batch]  # List of tensors
    xys = torch.stack([torch.from_numpy(item[2]) if isinstance(item[2], np.ndarray) 
                       else item[2] for item in batch])
    return images, bboxes, xys


def get_dataloader(
    labels_path: str,
    images_path: str,
    dataset: str = 'd1',
    split: str = 'train',
    input_size: int = 800,
    batch_size: int = 8,
    num_workers: int = 4,
    transform=None
):
    """Create a DataLoader for the DeepDarts dataset."""
    ds = DeepDartsDataset(
        labels_path=labels_path,
        images_path=images_path,
        dataset=dataset,
        split=split,
        input_size=input_size,
        transform=transform
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataloader
    labels_path = "deep-darts/dataset/labels.pkl"
    images_path = "deep-darts/dataset/cropped_images/800"
    
    # Check if images exist
    if not Path(images_path).exists():
        print(f"\n{'='*60}")
        print("DATASET NOT FOUND!")
        print("="*60)
        print(f"\nExpected images at: {images_path}")
        print("\nTo download the dataset:")
        print("1. Go to: https://ieee-dataport.org/open-access/deepdarts-dataset")
        print("2. Download 'cropped_images.zip' (3.35 GB)")
        print("3. Extract to: deep-darts/dataset/cropped_images/")
        print("="*60)
    else:
        print("Loading dataset...")
        loader = get_dataloader(
            labels_path=labels_path,
            images_path=images_path,
            dataset='d1',
            split='train',
            batch_size=4
        )
        
        # Get a batch
        images, bboxes, xys = next(iter(loader))
        print(f"\nBatch loaded successfully!")
        print(f"  Images shape: {images.shape}")
        print(f"  Bboxes: {[b.shape for b in bboxes]}")
        print(f"  XYs shape: {xys.shape}")
