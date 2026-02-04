"""
DeepDarts Inference Script
Demo script for predicting dart scores from images.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Add parent directory (Tipe) to path for imports
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from deepdarts.model.detector import DeepDartsDetector
from deepdarts.scoring.scorer import DartScorer
from deepdarts.utils.visualization import (
    visualize_predictions, 
    create_dartboard_image,
    draw_dartboard
)


class DeepDartsInference:
    """
    DeepDarts inference wrapper for easy prediction.
    """
    
    def __init__(self, model_path=None, device=None, input_size=416):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model weights (optional)
            device: Torch device to use
            input_size: Model input size
        """
        self.input_size = input_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create model
        self.model = DeepDartsDetector(num_classes=5, input_size=input_size)
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Using randomly initialized model (no weights loaded)")
        
        # Create scorer
        self.scorer = DartScorer()
    
    def preprocess(self, image):
        """Preprocess image for inference."""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original for visualization
        original = image.copy()
        h, w = image.shape[:2]
        
        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # To tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = image.unsqueeze(0)  # Add batch dimension
        
        return image, original, (h, w)
    
    def predict(self, image, conf_threshold=0.25):
        """
        Predict dart scores from an image.
        
        Args:
            image: Image path, numpy array (H, W, 3), or torch tensor
            conf_threshold: Confidence threshold for detections
            
        Returns:
            result: Dictionary with scores, points, and visualization
        """
        # Preprocess
        image_tensor, original, (orig_h, orig_w) = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            cal_points, dart_points, confidences = self.model.decode_predictions(
                outputs, conf_threshold=conf_threshold
            )
        
        # Scale points back to original image size
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        cal_points_orig = cal_points.cpu().numpy() * np.array([scale_x, scale_y])
        dart_points_orig = dart_points.cpu().numpy() * np.array([scale_x, scale_y]) if len(dart_points) > 0 else np.zeros((0, 2))
        
        # Calibrate and score
        success = self.scorer.calibrate(cal_points_orig)
        
        if success and len(dart_points_orig) > 0:
            scores, total = self.scorer.score_darts(dart_points_orig)
        else:
            scores, total = [], 0
        
        return {
            'calibration_points': cal_points_orig,
            'dart_points': dart_points_orig,
            'dart_confidences': confidences.cpu().numpy() if len(confidences) > 0 else np.array([]),
            'scores': scores,
            'total_score': total,
            'calibration_success': success,
            'original_image': original,
        }
    
    def visualize(self, result, save_path=None):
        """Create visualization of predictions."""
        fig = visualize_predictions(
            result['original_image'],
            result['calibration_points'],
            result['dart_points'],
            result['scores'],
            save_path=save_path
        )
        return fig


def demo_synthetic():
    """Run demo with synthetic dartboard image."""
    print("=" * 60)
    print("DeepDarts Demo - Synthetic Dartboard")
    print("=" * 60)
    
    # Create inference engine
    engine = DeepDartsInference()
    
    # Create synthetic dartboard
    print("\nGenerating synthetic dartboard...")
    image, cal_points = create_dartboard_image(400)
    
    # Add synthetic dart points
    dart_points = np.array([
        [200, 80],   # Near T20
        [200, 200],  # Bullseye area
        [280, 180],  # Right side
    ], dtype=np.float32)
    
    # Draw darts on image
    for pt in dart_points:
        cv2.circle(image, tuple(pt.astype(int)), 8, (255, 0, 255), -1)
        cv2.circle(image, tuple(pt.astype(int)), 10, (255, 255, 255), 2)
    
    # Run prediction
    print("Running inference...")
    result = engine.predict(image)
    
    print(f"\nResults:")
    print(f"  Calibration success: {result['calibration_success']}")
    print(f"  Darts detected: {len(result['dart_points'])}")
    print(f"  Individual scores: {result['scores']}")
    print(f"  Total score: {result['total_score']}")
    
    # Simulate correct scoring for demo
    print("\n--- Demo Mode ---")
    print("In demo mode with untrained model. Simulating scoring...")
    
    # Manual calibration with known points 
    engine.scorer.calibrate(cal_points)
    scores, total = engine.scorer.score_darts(dart_points)
    print(f"  Simulated scores: {scores}")
    print(f"  Simulated total: {total}")
    
    # Visualize
    output_path = Path("./demo_output.png")
    fig = visualize_predictions(
        image, 
        cal_points, 
        dart_points, 
        scores,
        save_path=str(output_path)
    )
    print(f"\nVisualization saved to: {output_path}")
    
    return result


def demo_dartboard_display():
    """Display a clean dartboard diagram."""
    print("\nGenerating dartboard diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    draw_dartboard(ax, center=(0, 0), radius=170)
    ax.set_title("Standard Dartboard", fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path("./dartboard_diagram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Dartboard diagram saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DeepDarts Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--output', type=str, default='output.png', help='Output path')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic image')
    parser.add_argument('--diagram', action='store_true', help='Generate dartboard diagram')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.diagram:
        demo_dartboard_display()
        return
    
    if args.demo or args.image is None:
        demo_synthetic()
        return
    
    # Run on provided image
    print(f"Processing: {args.image}")
    
    engine = DeepDartsInference(model_path=args.model)
    result = engine.predict(args.image, conf_threshold=args.conf)
    
    print(f"\nResults:")
    print(f"  Calibration success: {result['calibration_success']}")
    print(f"  Darts detected: {len(result['dart_points'])}")
    print(f"  Individual scores: {result['scores']}")
    print(f"  Total score: {result['total_score']}")
    
    # Visualize
    engine.visualize(result, save_path=args.output)
    print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
