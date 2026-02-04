import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Configuration
LABELS_PATH = 'deepdarts/data/cropped_images/cropped_images/labels.pkl'
OUTPUT_DIR = 'eda_output'

def run_eda():
    if not os.path.exists(LABELS_PATH):
        print(f"Error: Labels file not found at {LABELS_PATH}")
        return

    print(f"Loading labels from {LABELS_PATH}...")
    try:
        df = pd.read_pickle(LABELS_PATH)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        # Try finding it elsewhere if strict path fails? 
        # But this path is what train scripts use.
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset Composition
    print("Analyzing dataset composition...")
    
    # Define splits logic (copied from train_yolov8.py)
    D1_VAL = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
    D1_TEST = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 
               'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']
    D2_VAL = ['d2_02_03_2021', 'd2_02_05_2021']
    D2_TEST = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']
    
    val_folders = set(D1_VAL + D2_VAL)
    test_folders = set(D1_TEST + D2_TEST)
    
    def get_split(row):
        start_folder = row['img_folder']
        if start_folder in val_folders:
            return 'val'
        elif start_folder in test_folders:
            return 'test'
        return 'train'

    df['split'] = df.apply(get_split, axis=1)
    
    split_counts = df['split'].value_counts()
    print("\nDataset Split Counts:")
    print(split_counts)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=split_counts.index, y=split_counts.values, palette='viridis')
    plt.title('Number of Images per Split')
    plt.ylabel('Count')
    plt.savefig(f'{OUTPUT_DIR}/split_counts.png')
    plt.close()
    
    # 2. Darts per Image
    print("\nAnalyzing darts per image...")
    # indices 0-3 are calibration, 4+ are darts
    df['num_points'] = df['xy'].apply(lambda x: len(x))
    df['num_darts'] = df['num_points'].apply(lambda x: max(0, x - 4))
    
    dart_counts = df['num_darts'].value_counts().sort_index()
    print("\nDart Counts per Image Distribution:")
    print(dart_counts)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=dart_counts.index, y=dart_counts.values, palette='magma')
    plt.title('Distribution of Darts per Image')
    plt.xlabel('Number of Darts')
    plt.ylabel('Image Count')
    plt.savefig(f'{OUTPUT_DIR}/darts_per_image.png')
    plt.close()
    
    # 3. Spatial Heatmap
    print("\nGenerating spatial heatmap...")
    all_darts_x = []
    all_darts_y = []
    
    for _, row in df.iterrows():
        points = row['xy']
        if len(points) > 4:
            darts = points[4:] # Skip calibration
            for pt in darts:
                # Assuming normalized 0-1
                all_darts_x.append(pt[0])
                all_darts_y.append(pt[1])
                
    if all_darts_x:
        plt.figure(figsize=(10, 10))
        # Use 2D Histogram
        plt.hist2d(all_darts_x, all_darts_y, bins=100, cmap='hot', density=True)
        plt.colorbar(label='Density')
        plt.title('Spatial Density of Dart Landings')
        
        # Configure axes for image coordinates (0,0 top-left)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().invert_yaxis() 
        plt.gca().set_aspect('equal')
        
        plt.savefig(f'{OUTPUT_DIR}/dart_heatmap.png')
        plt.close()
        print(f"Processed {len(all_darts_x)} dart locations.")
    else:
        print("No darts found to plot.")

    # 4. Calibration Points Spread (Optional interesting metric)
    print("\nAnalyzing calibration points...")
    cal_x = []
    cal_y = []
    for _, row in df.iterrows():
        points = row['xy']
        if len(points) >= 4:
            # First 4 are calibration
            # Usually: Top, Right, Bottom, Left order? 
            # Let's just plot all of them to see the grid
            for pt in points[:4]:
                cal_x.append(pt[0])
                cal_y.append(pt[1])
                
    if cal_x:
        plt.figure(figsize=(8, 8))
        plt.scatter(cal_x, cal_y, alpha=0.1, s=1)
        plt.title('Calibration Point Locations')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.savefig(f'{OUTPUT_DIR}/calibration_scatter.png')
        plt.close()

    print(f"\nEDA Complete. Results saved to directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_eda()
