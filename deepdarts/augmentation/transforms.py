"""
Data Augmentation for DeepDarts
Task-specific augmentations from the paper.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional


class DartboardFlip:
    """
    Randomly flip dartboard image and dart positions.
    Calibration points remain fixed.
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        """
        Args:
            image: Image tensor (C, H, W) or numpy (H, W, C)
            calibration_points: (4, 2) array
            dart_points: (N, 2) array
            H: Optional homography matrix
            
        Returns:
            Augmented image, calibration_points, dart_points
        """
        if np.random.random() > self.p:
            return image, calibration_points, dart_points
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
        
        h, w = image.shape[:2]
        
        # Random flip type
        flip_type = np.random.choice(['h', 'v', 'hv'])
        
        if 'h' in flip_type:
            image = np.fliplr(image).copy()
            if dart_points is not None and len(dart_points) > 0:
                dart_points = dart_points.copy()
                dart_points[:, 0] = w - dart_points[:, 0]
        
        if 'v' in flip_type:
            image = np.flipud(image).copy()
            if dart_points is not None and len(dart_points) > 0:
                dart_points = dart_points.copy()
                dart_points[:, 1] = h - dart_points[:, 1]
        
        if is_tensor:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        return image, calibration_points, dart_points


class DartboardRotation:
    """
    Rotate dartboard image and keypoints by multiples of 18° or 36°.
    This keeps dartboard sections aligned.
    """
    
    def __init__(self, step_degrees=36, p=0.5):
        """
        Args:
            step_degrees: 18 or 36 degrees
            p: Probability of applying augmentation
        """
        self.step_degrees = step_degrees
        self.p = p
        self.num_steps = 360 // step_degrees
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        if np.random.random() > self.p:
            return image, calibration_points, dart_points
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Random rotation angle (multiples of step_degrees)
        step = np.random.randint(-self.num_steps // 2, self.num_steps // 2 + 1)
        angle = step * self.step_degrees
        
        if angle == 0:
            if is_tensor:
                image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
            return image, calibration_points, dart_points
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Rotate image
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        
        # Rotate dart points (calibration points stay fixed for consistent detection)
        if dart_points is not None and len(dart_points) > 0:
            dart_points = self._rotate_points(dart_points, M)
        
        if is_tensor:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        return image, calibration_points, dart_points
    
    def _rotate_points(self, points, M):
        """Apply rotation matrix to points."""
        points = points.copy()
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])
        rotated = (M @ points_h.T).T
        return rotated


class SmallRotation:
    """
    Apply small random rotations (-2° to 2°) to account for
    dartboards that are not perfectly vertically aligned.
    """
    
    def __init__(self, max_degrees=2, p=0.5):
        self.max_degrees = max_degrees
        self.p = p
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        if np.random.random() > self.p:
            return image, calibration_points, dart_points
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Random small angle
        angle = np.random.uniform(-self.max_degrees, self.max_degrees)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Rotate everything (including calibration points)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        
        if calibration_points is not None and len(calibration_points) > 0:
            calibration_points = self._rotate_points(calibration_points, M)
        
        if dart_points is not None and len(dart_points) > 0:
            dart_points = self._rotate_points(dart_points, M)
        
        if is_tensor:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        return image, calibration_points, dart_points
    
    def _rotate_points(self, points, M):
        points = points.copy()
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])
        rotated = (M @ points_h.T).T
        return rotated


class PerspectiveWarping:
    """
    Randomly warp perspective to generalize to various camera angles.
    This is the most effective augmentation for the D2 dataset.
    """
    
    def __init__(self, rho=0.5, p=0.5):
        """
        Args:
            rho: Maximum scaling factor for homography perturbation
            p: Probability of applying augmentation
        """
        self.rho = rho
        self.p = p
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        if np.random.random() > self.p:
            return image, calibration_points, dart_points
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        h, w = image.shape[:2]
        
        # Create random perspective transformation
        # Perturb corners
        margin = min(h, w) * self.rho * 0.1
        
        src_pts = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Random perturbations
        dst_pts = src_pts + np.random.uniform(-margin, margin, src_pts.shape).astype(np.float32)
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply to image
        image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        
        # Apply to points
        if calibration_points is not None and len(calibration_points) > 0:
            calibration_points = self._warp_points(calibration_points, M)
        
        if dart_points is not None and len(dart_points) > 0:
            dart_points = self._warp_points(dart_points, M)
        
        if is_tensor:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        
        return image, calibration_points, dart_points
    
    def _warp_points(self, points, M):
        points = points.copy().astype(np.float32)
        ones = np.ones((len(points), 1), dtype=np.float32)
        points_h = np.hstack([points, ones])
        warped_h = (M @ points_h.T).T
        warped = warped_h[:, :2] / warped_h[:, 2:3]
        return warped


class ColorJitter:
    """Apply color jittering for lighting variations."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        if np.random.random() > self.p:
            return image, calibration_points, dart_points
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.float() / 255.0
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Brightness
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            image = image * factor
        
        # Contrast
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            mean = image.mean() if isinstance(image, np.ndarray) else image.mean().item()
            image = (image - mean) * factor + mean
        
        # Clip values
        image = np.clip(image, 0, 1) if isinstance(image, np.ndarray) else image.clamp(0, 1)
        
        # Convert back
        if is_tensor:
            image = (image.permute(2, 0, 1) * 255).byte()
        else:
            image = (image * 255).astype(np.uint8)
        
        return image, calibration_points, dart_points


class Compose:
    """Compose multiple augmentations."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, calibration_points, dart_points, H=None):
        for t in self.transforms:
            image, calibration_points, dart_points = t(
                image, calibration_points, dart_points, H
            )
        return image, calibration_points, dart_points


def get_train_transforms(p=0.8, step_degrees=36):
    """Get default training augmentations."""
    return Compose([
        DartboardFlip(p=p * 0.5),
        DartboardRotation(step_degrees=step_degrees, p=p * 0.5),
        SmallRotation(max_degrees=2, p=p * 0.5),
        PerspectiveWarping(rho=0.5, p=p * 0.5),
        ColorJitter(p=p * 0.5),
    ])


if __name__ == "__main__":
    # Test augmentations
    import matplotlib.pyplot as plt
    
    # Create a synthetic dartboard image (simple colored regions)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(image, (200, 200), 180, (50, 50, 50), -1)
    cv2.circle(image, (200, 200), 150, (255, 255, 255), -1)
    cv2.circle(image, (200, 200), 100, (50, 50, 50), -1)
    cv2.circle(image, (200, 200), 50, (0, 255, 0), -1)
    cv2.circle(image, (200, 200), 20, (0, 0, 255), -1)
    
    # Calibration points (corners of a square)
    cal_points = np.array([
        [200, 30],
        [370, 200],
        [200, 370],
        [30, 200],
    ], dtype=np.float32)
    
    # Dart points
    dart_points = np.array([
        [200, 100],
        [250, 200],
    ], dtype=np.float32)
    
    # Draw points
    for i, pt in enumerate(cal_points):
        cv2.circle(image, tuple(pt.astype(int)), 8, (255, 255, 0), -1)
    for pt in dart_points:
        cv2.circle(image, tuple(pt.astype(int)), 8, (255, 0, 255), -1)
    
    # Apply augmentations
    transforms = get_train_transforms(p=1.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    
    for i in range(1, 6):
        aug_img, aug_cal, aug_dart = transforms(image.copy(), cal_points.copy(), dart_points.copy())
        
        # Draw augmented points
        for pt in aug_cal:
            cv2.circle(aug_img, tuple(pt.astype(int)), 8, (255, 255, 0), -1)
        for pt in aug_dart:
            cv2.circle(aug_img, tuple(pt.astype(int)), 8, (255, 0, 255), -1)
        
        ax = axes[i // 3, i % 3]
        ax.imshow(aug_img)
        ax.set_title(f"Augmented {i}")
    
    plt.tight_layout()
    plt.savefig("augmentation_test.png")
    print("Saved augmentation_test.png")
