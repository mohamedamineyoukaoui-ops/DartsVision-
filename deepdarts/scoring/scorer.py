"""
Dart Scoring Module
Calculates dart scores based on position on dartboard.
"""

import numpy as np
import torch
from .homography import DartboardCalibrator


class DartScorer:
    """
    Calculates dart scores based on their position on the dartboard.
    Uses polar coordinates to classify darts into sections.
    """
    
    # Dartboard section order (clockwise from top)
    SECTIONS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    
    # Dartboard radii (in mm, from center)
    # Based on British Darts Organisation specifications
    RADII = {
        'double_bull': 6.35,      # Inner bullseye (50 points)
        'bull': 15.9,             # Outer bullseye (25 points)
        'inner_single': 99.0,     # Inner single section
        'treble': 107.0,          # Treble ring (outer edge)
        'outer_single': 162.0,    # Outer single section
        'double': 170.0,          # Double ring (outer edge)
    }
    
    def __init__(self, outer_radius=170.0):
        """
        Args:
            outer_radius: Outer edge radius used for normalization
        """
        self.outer_radius = outer_radius
        self.calibrator = DartboardCalibrator(outer_radius)
        
        # Precompute section boundaries (in radians)
        self.section_angles = self._compute_section_angles()
    
    def _compute_section_angles(self):
        """Compute angle boundaries for each section."""
        angles = {}
        section_width = 2 * np.pi / 20  # 18 degrees per section
        
        for i, section in enumerate(self.SECTIONS):
            # Center angle (section 20 is at top, angle 0)
            center = i * section_width
            start = center - section_width / 2
            end = center + section_width / 2
            angles[section] = (start, end)
        
        return angles
    
    def calibrate(self, calibration_points):
        """
        Calibrate the scorer with detected calibration points.
        
        Args:
            calibration_points: Detected calibration points (4, 2) in image coords
            
        Returns:
            success: Whether calibration was successful
        """
        return self.calibrator.calibrate(calibration_points)
    
    def score_darts(self, dart_points):
        """
        Calculate scores for all darts.
        
        Args:
            dart_points: Dart positions in image coordinates (N, 2)
            
        Returns:
            scores: List of individual dart scores
            total: Total score
        """
        if isinstance(dart_points, torch.Tensor):
            dart_points = dart_points.cpu().numpy()
        
        if len(dart_points) == 0:
            return [], 0
        
        # Transform to dartboard coordinates
        transformed = self.calibrator.transform_to_dartboard(dart_points)
        
        # Score each dart
        scores = []
        for point in transformed:
            score = self._score_single_dart(point)
            scores.append(score)
        
        return scores, sum(scores)
    
    def _score_single_dart(self, point):
        """
        Calculate score for a single dart.
        
        Args:
            point: Dart position in dartboard coordinates (2,)
            
        Returns:
            score: Integer score
        """
        # Calculate polar coordinates relative to center
        center = self.calibrator.center
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dx, -dy)  # Angle from top, clockwise
        
        # Normalize angle to [0, 2*pi)
        if angle < 0:
            angle += 2 * np.pi
        
        # Determine region based on radius
        region = self._get_region(radius)
        
        if region == 'double_bull':
            return 50
        elif region == 'bull':
            return 25
        elif region == 'miss':
            return 0
        
        # Get section number based on angle
        section = self._get_section(angle)
        
        # Calculate final score based on region
        if region == 'treble':
            return section * 3
        elif region == 'double':
            return section * 2
        else:  # Single
            return section
    
    def _get_region(self, radius):
        """Determine dartboard region based on radius."""
        # Scale radii based on calibrated radius
        scale = self.calibrator.radius / self.outer_radius
        
        if radius <= self.RADII['double_bull'] * scale:
            return 'double_bull'
        elif radius <= self.RADII['bull'] * scale:
            return 'bull'
        elif radius <= self.RADII['inner_single'] * scale:
            return 'inner_single'
        elif radius <= self.RADII['treble'] * scale:
            return 'treble'
        elif radius <= self.RADII['outer_single'] * scale:
            return 'outer_single'
        elif radius <= self.RADII['double'] * scale:
            return 'double'
        else:
            return 'miss'
    
    def _get_section(self, angle):
        """Determine section number based on angle."""
        # Each section is 18 degrees (pi/10 radians)
        section_width = 2 * np.pi / 20
        
        # Find which section the angle falls into
        # Section 20 is centered at angle 0
        section_idx = int((angle + section_width / 2) / section_width) % 20
        
        return self.SECTIONS[section_idx]
    
    def get_score_name(self, score, section=None, region=None):
        """
        Get human-readable name for a score.
        
        Args:
            score: The point value
            section: Optional section number
            region: Optional region name
            
        Returns:
            name: Human-readable score name (e.g., "T20", "D16", "25")
        """
        if score == 50:
            return "DB"  # Double Bull
        elif score == 25:
            return "B"   # Bull
        elif score == 0:
            return "Miss"
        
        if section is None:
            # Try to infer section and multiplier
            for mult, prefix in [(3, 'T'), (2, 'D'), (1, '')]:
                if score % mult == 0 and score // mult <= 20:
                    section = score // mult
                    return f"{prefix}{section}"
        
        if region == 'treble':
            return f"T{section}"
        elif region == 'double':
            return f"D{section}"
        else:
            return str(score)


class DeepDartsScorer:
    """
    Complete DeepDarts scoring pipeline.
    Combines keypoint detection with score calculation.
    """
    
    def __init__(self, model=None):
        """
        Args:
            model: DeepDartsDetector model (optional, can be set later)
        """
        self.model = model
        self.scorer = DartScorer()
    
    def predict(self, image):
        """
        Predict dart scores from an image.
        
        Args:
            image: Input image tensor (3, H, W) or (B, 3, H, W)
            
        Returns:
            scores: List of individual dart scores
            total: Total score
            calibration_points: Detected calibration points
            dart_points: Detected dart positions
        """
        if self.model is None:
            raise ValueError("Model not set. Set self.model before calling predict().")
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image)
            cal_points, dart_points, confidences = self.model.decode_predictions(outputs)
        
        # Calibrate
        success = self.scorer.calibrate(cal_points)
        
        if not success:
            return [], 0, cal_points, dart_points
        
        # Score darts
        scores, total = self.scorer.score_darts(dart_points)
        
        return scores, total, cal_points, dart_points


# Convenience function for Percent Correct Score (PCS) metric
def compute_pcs(predictions, targets):
    """
    Compute Percent Correct Score metric.
    
    Args:
        predictions: List of predicted total scores
        targets: List of ground truth total scores
        
    Returns:
        pcs: Percentage of correct total score predictions
    """
    if len(predictions) != len(targets):
        raise ValueError("Length mismatch between predictions and targets")
    
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return 100.0 * correct / len(predictions) if len(predictions) > 0 else 0.0


if __name__ == "__main__":
    # Test the scorer with synthetic data
    
    # Create synthetic calibration points (square dartboard view)
    cal_points = np.array([
        [200, 50],   # Top (5-20 intersection)
        [350, 200],  # Right (13-6 intersection)
        [200, 350],  # Bottom (17-3 intersection)
        [50, 200],   # Left (8-11 intersection)
    ], dtype=np.float32)
    
    # Create scorer and calibrate
    scorer = DartScorer()
    success = scorer.calibrate(cal_points)
    print(f"Calibration success: {success}")
    
    # Test with some dart positions
    dart_points = np.array([
        [200, 75],   # Near top (should be 20 area)
        [200, 200],  # Center (should be bullseye)
        [300, 200],  # Right side
    ], dtype=np.float32)
    
    scores, total = scorer.score_darts(dart_points)
    print(f"\nDart scores: {scores}")
    print(f"Total score: {total}")
    
    # Test PCS metric
    predictions = [60, 100, 45, 180]
    targets = [60, 100, 50, 180]
    pcs = compute_pcs(predictions, targets)
    print(f"\nPCS: {pcs:.1f}% (expected 75.0%)")
