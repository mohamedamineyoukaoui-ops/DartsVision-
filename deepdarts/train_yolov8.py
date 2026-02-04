"""
DeepDarts YOLOv8 Training Script
Uses state-of-the-art YOLOv8 for keypoint/object detection.
NO GPU LIMITS - uses all available VRAM.
"""

import os
import sys
import shutil
import multiprocessing
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

# Add parent directory
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def prepare_yolo_dataset(labels_path: str, images_path: str, output_dir: str, dataset: str = 'd1'):
    """
    Convert DeepDarts dataset to YOLO format.
    
    YOLO format: class x_center y_center width height (all normalized 0-1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Dataset splits
    D1_VAL = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
    D1_TEST = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 
               'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']
    D2_VAL = ['d2_02_03_2021', 'd2_02_05_2021']
    D2_TEST = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']
    
    if dataset == 'd1':
        val_folders, test_folders = D1_VAL, D1_TEST
    else:
        val_folders, test_folders = D2_VAL, D2_TEST
    
    # Load labels
    df = pd.read_pickle(labels_path)
    df = df[df.img_folder.str.contains(dataset)]
    
    images_path = Path(images_path)
    bbox_size = 0.025  # Bounding box size as fraction of image
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    for idx, row in df.iterrows():
        # Determine split
        if row['img_folder'] in val_folders:
            split = 'val'
        elif row['img_folder'] in test_folders:
            split = 'test'
        else:
            split = 'train'
        
        # Source image path
        src_img = images_path / row['img_folder'] / row['img_name']
        if not src_img.exists():
            continue
        
        # Create unique filename
        img_name = f"{row['img_folder']}_{row['img_name']}"
        dst_img = output_dir / 'images' / split / img_name
        
        # Copy image (or symlink for speed)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        
        # Create YOLO label file
        label_file = output_dir / 'labels' / split / (Path(img_name).stem + '.txt')
        
        xy = np.array(row['xy'])  # Shape: (N, 2) where N <= 7
        
        with open(label_file, 'w') as f:
            for i, pt in enumerate(xy):
                x, y = pt[0], pt[1]
                
                # Skip if too close to edge
                half = bbox_size / 2
                if x - half <= 0 or x + half >= 1 or y - half <= 0 or y + half >= 1:
                    continue
                
                # Class: 0 for darts (indices 4-6), 1-4 for calibration (indices 0-3)
                if i < 4:
                    cls = i + 1  # Calibration points are classes 1-4
                else:
                    cls = 0  # Darts are class 0
                
                # YOLO format: class x_center y_center width height
                f.write(f"{cls} {x:.6f} {y:.6f} {bbox_size:.6f} {bbox_size:.6f}\n")
        
        counts[split] += 1
    
    print(f"Dataset prepared: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,
        'names': {
            0: 'dart',
            1: 'cal_top',
            2: 'cal_right',
            3: 'cal_bottom',
            4: 'cal_left'
        }
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created {yaml_path}")
    return str(yaml_path)


def train_yolov8(
    data_yaml: str,
    model_size: str = 'l',  # Default to Large for best accuracy
    epochs: int = 300,      # Increase epochs for best convergence
    batch_size: int = -1,   # -1 = auto-detect max batch for GPU
    imgsz: int = 800,
    output_dir: str = 'output_yolov8'
):
    """
    Train YOLOv8 on DeepDarts dataset.
    
    Args:
        data_yaml: Path to data.yaml
        model_size: 'n', 's', 'm', 'l', or 'x' (nano to extra-large)
        epochs: Number of training epochs
        batch_size: Batch size (-1 for auto-detect maximum)
        imgsz: Image size
        output_dir: Output directory
    """
    print("=" * 70)
    print(f"YOLOv8{model_size} Training - Optimization: BEST MODEL + MAX SPEED")
    print("=" * 70)
    
    # Load pretrained YOLOv8
    # 'l' (large) or 'x' (extra-large) recommended for "best possible model"
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Get max CPU workers
    num_workers = multiprocessing.cpu_count()
    
    # Train with MAX PERFORMANCE settings
    # MATH FOR 8GB VRAM @ 800px IMG SIZE (YOLOv8-Large):
    # - Base Model Overhead: ~2.5 GB
    # - Cost per Image (800x800, FP16): ~1.5 GB
    # - Max Batch = (8.0 - 2.5) / 1.5 = ~3.6 -> Round down to Safe Max = 3
    # - Batch 4 is likely to OOM (Out Of Memory).
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,   # Use the passed batch size!
        project=output_dir,
        name=f'deepdarts_{model_size}_best',
        
        # Optimization for ACCURACY ("Best Model")
        optimizer='auto',   # Valid choices: SGD, Adam, Adamax, AdamW, NAdam, RAdam or auto
        cos_lr=True,        # Cosine learning rate scheduler
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final learning rate fraction
        momentum=0.937,     # Momentum
        weight_decay=0.0005, # Weight decay
        warmup_epochs=3.0,  # Warmup epochs
        box=7.5,            # Box loss gain
        cls=0.5,            # Class loss gain
        dfl=1.5,            # DFL loss gain
        
        # Augmentation (Stronger for better generalization)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,        # Increased rotation
        translate=0.1,
        scale=0.6,           # Increased scaling
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,           # Added mixup
        copy_paste=0.1,      # Added copy-paste
        
        # Training Strategy
        close_mosaic=20,     # Disable mosaic for last 20 epochs
        patience=50,         # Early stopping
        
        # Hardware Speed Optimizations ("Fast as PC can")
        device=0,            # Use GPU 0
        workers=8,           # Maximize CPU usage
        cache=True,          # Enable RAM cache for speed (User approved)
        amp=True,            # Automatic Mixed Precision
        
        # Misc
        verbose=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        exist_ok=True
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DeepDarts')
    parser.add_argument('--model', type=str, default='s', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n(nano), s(small), m(medium), l(large), x(extra-large)')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size (-1 for auto-detect max)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--dataset', type=str, default='d1', help='Dataset: d1 or d2')
    parser.add_argument('--data-path', type=str, 
                       default='deepdarts/data/cropped_images/cropped_images/800',
                       help='Path to images')
    parser.add_argument('--labels-path', type=str,
                       default='deepdarts/data/cropped_images/cropped_images/labels.pkl',
                       help='Path to labels.pkl')
    parser.add_argument('--output', type=str, default='output_yolov8',
                       help='Output directory')
    parser.add_argument('--skip-prep', action='store_true',
                       help='Skip dataset preparation (if already done)')
    
    args = parser.parse_args()
    
    # Prepare dataset
    yolo_data_dir = Path('deepdarts_yolo_data')
    yaml_path = yolo_data_dir / 'data.yaml'
    
    if not args.skip_prep or not yaml_path.exists():
        print("\n📦 Preparing YOLO format dataset...")
        yaml_path = prepare_yolo_dataset(
            labels_path=args.labels_path,
            images_path=args.data_path,
            output_dir=str(yolo_data_dir),
            dataset=args.dataset
        )
    else:
        yaml_path = str(yaml_path)
        print(f"✓ Using existing dataset: {yaml_path}")
    
    # Train
    print(f"\n🚀 Training YOLOv8{args.model}...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {'Auto-max' if args.batch == -1 else args.batch}")
    print(f"   Image size: {args.imgsz}")
    
    train_yolov8(
        data_yaml=yaml_path,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
