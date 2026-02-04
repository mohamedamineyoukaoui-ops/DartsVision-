"""
Visualization utilities for DeepDarts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge
import cv2


# Dartboard colors
COLORS = {
    'black': '#1a1a1a',
    'white': '#f5f5dc',
    'red': '#c41e3a',
    'green': '#228b22',
    'gold': '#ffd700',
}

# Section order (clockwise from top)
SECTIONS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def draw_dartboard(ax, center=(0, 0), radius=170, show_numbers=True):
    """
    Draw a dartboard on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis
        center: Center coordinates (x, y)
        radius: Outer radius in mm
    """
    cx, cy = center
    
    # Radii (relative to outer double ring)
    r_double = radius
    r_outer_single = 162 * radius / 170
    r_treble = 107 * radius / 170
    r_inner_single = 99 * radius / 170
    r_bull = 15.9 * radius / 170
    r_double_bull = 6.35 * radius / 170
    
    # Draw sections
    section_angle = 360 / 20
    
    for i, section in enumerate(SECTIONS):
        start_angle = 90 - (i + 0.5) * section_angle
        
        # Alternate colors
        if i % 2 == 0:
            single_color = COLORS['black']
            ring_color = COLORS['red']
        else:
            single_color = COLORS['white']
            ring_color = COLORS['green']
        
        # Outer single
        wedge = Wedge(center, r_outer_single, start_angle, start_angle + section_angle,
                     width=r_outer_single - r_treble, facecolor=single_color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(wedge)
        
        # Double ring
        wedge = Wedge(center, r_double, start_angle, start_angle + section_angle,
                     width=r_double - r_outer_single, facecolor=ring_color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(wedge)
        
        # Treble ring
        wedge = Wedge(center, r_treble, start_angle, start_angle + section_angle,
                     width=r_treble - r_inner_single, facecolor=ring_color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(wedge)
        
        # Inner single
        wedge = Wedge(center, r_inner_single, start_angle, start_angle + section_angle,
                     width=r_inner_single - r_bull, facecolor=single_color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(wedge)
    
    # Bull (outer bullseye)
    bull = Circle(center, r_bull, facecolor=COLORS['green'], edgecolor='gray')
    ax.add_patch(bull)
    
    # Double bull (inner bullseye)
    double_bull = Circle(center, r_double_bull, facecolor=COLORS['red'], edgecolor='gray')
    ax.add_patch(double_bull)
    
    # Add numbers
    if show_numbers:
        for i, section in enumerate(SECTIONS):
            angle = np.radians(90 - i * section_angle)
            x = cx + (radius + 15) * np.cos(angle)
            y = cy + (radius + 15) * np.sin(angle)
            ax.text(x, y, str(section), ha='center', va='center', 
                   fontsize=10, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_xlim(cx - radius - 30, cx + radius + 30)
    ax.set_ylim(cy - radius - 30, cy + radius + 30)
    ax.axis('off')


def visualize_predictions(image, calibration_points, dart_points, scores=None, save_path=None):
    """
    Visualize predictions on an image.
    
    Args:
        image: Input image (H, W, 3) or (3, H, W)
        calibration_points: (4, 2) array of calibration points
        dart_points: (N, 2) array of dart positions
        scores: Optional list of scores for each dart
        save_path: Optional path to save the figure
    """
    if isinstance(image, np.ndarray):
        if image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
    else:
        import torch
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Draw calibration points
    cal_colors = ['#FFD700', '#FF6B00', '#00FF00', '#00BFFF']  # Gold, Orange, Green, Blue
    cal_labels = ['5-20', '13-6', '17-3', '8-11']
    
    for i, (pt, color, label) in enumerate(zip(calibration_points, cal_colors, cal_labels)):
        ax.scatter(pt[0], pt[1], c=color, s=200, marker='s', edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(label, (pt[0], pt[1]), xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Draw dart points
    for i, pt in enumerate(dart_points):
        ax.scatter(pt[0], pt[1], c='#FF00FF', s=300, marker='o', edgecolors='white', linewidths=2, zorder=6)
        
        if scores is not None and i < len(scores):
            ax.annotate(str(scores[i]), (pt[0], pt[1]), xytext=(15, 15), textcoords='offset points',
                       fontsize=12, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FF00FF', alpha=0.9))
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        print(f"Saved to {save_path}")
    
    return fig


def visualize_training_history(history, save_path=None):
    """
    Plot training history curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_pcs', 'val_pcs'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCS plot
    ax2 = axes[1]
    if 'train_pcs' in history:
        ax2.plot(history['train_pcs'], 'b-', label='Train PCS', linewidth=2)
    if 'val_pcs' in history:
        ax2.plot(history['val_pcs'], 'r-', label='Val PCS', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Percent Correct Score (%)', fontsize=12)
    ax2.set_title('Training & Validation PCS', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    
    return fig


def create_dartboard_image(size=400, background='white'):
    """
    Create a synthetic dartboard image for testing.
    
    Args:
        size: Image size (square)
        background: Background color
        
    Returns:
        image: RGB image (size, size, 3)
        calibration_points: (4, 2) array
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    if background == 'white':
        image[:] = 255
    
    center = size // 2
    radius = int(size * 0.4)
    
    # Draw dartboard
    section_angle = 18  # degrees
    
    for i in range(20):
        start = -90 - section_angle / 2 + i * section_angle
        end = start + section_angle
        
        if i % 2 == 0:
            color1 = (26, 26, 26)  # Black
            color2 = (58, 30, 196)  # Red (BGR)
        else:
            color1 = (245, 245, 220)  # Beige
            color2 = (34, 139, 34)  # Green (BGR)
        
        # Draw double ring
        cv2.ellipse(image, (center, center), (radius, radius),
                   0, start, end, color2[::-1], -1)
        
        # Draw outer single
        r2 = int(radius * 0.95)
        cv2.ellipse(image, (center, center), (r2, r2),
                   0, start, end, color1[::-1], -1)
        
        # Draw treble ring
        r3 = int(radius * 0.63)
        cv2.ellipse(image, (center, center), (r3, r3),
                   0, start, end, color2[::-1], -1)
        
        # Draw inner single
        r4 = int(radius * 0.58)
        cv2.ellipse(image, (center, center), (r4, r4),
                   0, start, end, color1[::-1], -1)
    
    # Draw bull
    cv2.circle(image, (center, center), int(radius * 0.1), (34, 139, 34)[::-1], -1)
    
    # Draw double bull
    cv2.circle(image, (center, center), int(radius * 0.04), (58, 30, 196)[::-1], -1)
    
    # Calibration points at 4 cardinal directions on outer edge
    cal_points = np.array([
        [center, center - radius],  # Top (5-20)
        [center + radius, center],  # Right (13-6)
        [center, center + radius],  # Bottom (17-3)
        [center - radius, center],  # Left (8-11)
    ], dtype=np.float32)
    
    return image, cal_points


if __name__ == "__main__":
    # Test visualization
    
    # Create synthetic dartboard
    image, cal_points = create_dartboard_image(400)
    
    # Add some dart points
    dart_points = np.array([
        [200, 100],  # Near T20
        [200, 200],  # Bullseye
        [250, 180],  # Single area
    ], dtype=np.float32)
    
    scores = [60, 50, 5]
    
    # Visualize
    fig = visualize_predictions(image, cal_points, dart_points, scores, save_path="prediction_viz.png")
    
    # Draw isolated dartboard
    fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
    draw_dartboard(ax)
    plt.savefig("dartboard_viz.png", dpi=150)
    print("Saved dartboard_viz.png")
    
    # Test training history plot
    history = {
        'train_loss': np.exp(-np.linspace(0, 2, 50)) + 0.1 + np.random.randn(50) * 0.02,
        'val_loss': np.exp(-np.linspace(0, 1.5, 50)) + 0.15 + np.random.randn(50) * 0.03,
        'train_pcs': 50 + 40 * (1 - np.exp(-np.linspace(0, 3, 50))) + np.random.randn(50) * 2,
        'val_pcs': 45 + 35 * (1 - np.exp(-np.linspace(0, 2.5, 50))) + np.random.randn(50) * 3,
    }
    visualize_training_history(history, save_path="training_history.png")
