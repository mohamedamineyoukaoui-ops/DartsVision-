"""
Homography calculation for dartboard coordinate transformation.
Transforms image coordinates to dartboard plane.
"""

import numpy as np
import torch


def compute_homography(src_points, dst_points):
    """
    Compute homography matrix using Direct Linear Transform (DLT).
    
    Args:
        src_points: Source points (4, 2) - calibration points in image
        dst_points: Destination points (4, 2) - known positions on dartboard
        
    Returns:
        H: Homography matrix (3, 3)
    """
    if isinstance(src_points, torch.Tensor):
        src_points = src_points.cpu().numpy()
    if isinstance(dst_points, torch.Tensor):
        dst_points = dst_points.cpu().numpy()
    
    assert src_points.shape == (4, 2), f"Expected 4 source points, got {src_points.shape}"
    assert dst_points.shape == (4, 2), f"Expected 4 destination points, got {dst_points.shape}"
    
    # Build the DLT matrix A
    A = []
    for i in range(4):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A, dtype=np.float64)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H


def transform_points(points, H):
    """
    Transform points using homography matrix.
    
    Args:
        points: Points to transform (N, 2)
        H: Homography matrix (3, 3)
        
    Returns:
        Transformed points (N, 2)
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to homogeneous coordinates
    N = points.shape[0]
    ones = np.ones((N, 1))
    points_h = np.hstack([points, ones])  # (N, 3)
    
    # Apply homography
    transformed_h = (H @ points_h.T).T  # (N, 3)
    
    # Convert back from homogeneous
    transformed = transformed_h[:, :2] / transformed_h[:, 2:3]
    
    return transformed


class DartboardCalibrator:
    """
    Calibrates dartboard from detected calibration points.
    
    The 4 calibration points are at the outer edge of the double ring,
    at intersections: (5,20), (13,6), (17,3), (8,11)
    """
    
    # Known calibration point positions on normalized dartboard (radius=1)
    # Angles for each calibration point (in radians from top, clockwise)
    SECTION_ANGLES = {
        20: 0, 1: 18, 18: 36, 4: 54, 13: 72,
        6: 90, 10: 108, 15: 126, 2: 144, 17: 162,
        3: 180, 19: 198, 7: 216, 16: 234, 8: 252,
        11: 270, 14: 288, 9: 306, 12: 324, 5: 342
    }
    
    # Calibration points are at these section intersections
    CAL_SECTIONS = [(5, 20), (13, 6), (17, 3), (8, 11)]
    
    def __init__(self, outer_radius=170.0):
        """
        Args:
            outer_radius: Radius of the outer double ring edge (mm)
        """
        self.outer_radius = outer_radius
        self.target_points = self._compute_target_calibration_points()
        self.H = None
        self.center = None
        self.radius = None
    
    def _compute_target_calibration_points(self):
        """Compute the target calibration point positions in normalized coords."""
        points = []
        for s1, s2 in self.CAL_SECTIONS:
            # Calibration point is between two sections
            angle1 = np.radians(self.SECTION_ANGLES[s1])
            angle2 = np.radians(self.SECTION_ANGLES[s2])
            
            # Average angle (handle wraparound)
            if abs(angle1 - angle2) > np.pi:
                angle = (angle1 + angle2 + 2 * np.pi) / 2
                if angle > 2 * np.pi:
                    angle -= 2 * np.pi
            else:
                angle = (angle1 + angle2) / 2
            
            # Point at outer edge
            x = self.outer_radius * np.sin(angle)
            y = -self.outer_radius * np.cos(angle)  # Negative because y-axis is flipped
            points.append([x, y])
        
        return np.array(points)
    
    def calibrate(self, calibration_points):
        """
        Calibrate the dartboard using detected calibration points.
        
        Args:
            calibration_points: Detected calibration points (4, 2) in image coords
            
        Returns:
            success: Whether calibration was successful
        """
        if isinstance(calibration_points, torch.Tensor):
            calibration_points = calibration_points.cpu().numpy()
        
        # Check for missing calibration points
        valid_mask = np.all(calibration_points != 0, axis=1)
        
        if valid_mask.sum() < 3:
            return False
        
        if valid_mask.sum() == 3:
            # Estimate missing point from other three
            missing_idx = np.where(~valid_mask)[0][0]
            calibration_points = self._estimate_missing_point(
                calibration_points, missing_idx
            )
        
        # Compute homography
        self.H = compute_homography(calibration_points, self.target_points)
        
        # Transform calibration points to verify
        transformed = transform_points(calibration_points, self.H)
        
        # Compute center and radius
        self.center = np.mean(transformed, axis=0)
        distances = np.linalg.norm(transformed - self.center, axis=1)
        self.radius = np.mean(distances)
        
        return True
    
    def _estimate_missing_point(self, points, missing_idx):
        """Estimate a missing calibration point from the other three."""
        # Use the fact that calibration points form a quadrilateral
        # The missing point can be estimated from opposite point and two adjacent
        
        # Simple approach: use center and distance estimate
        valid_points = points[points.sum(axis=1) != 0]
        center = np.mean(valid_points, axis=0)
        
        # Estimate based on opposite point
        opposite_idx = (missing_idx + 2) % 4
        if np.all(points[opposite_idx] != 0):
            # Mirror through center
            points[missing_idx] = 2 * center - points[opposite_idx]
        else:
            # Fallback: use average position
            points[missing_idx] = center
        
        return points
    
    def transform_to_dartboard(self, points):
        """
        Transform image points to dartboard coordinates.
        
        Args:
            points: Points in image coordinates (N, 2)
            
        Returns:
            Transformed points in dartboard coordinates (N, 2)
        """
        if self.H is None:
            raise ValueError("Calibration not computed. Call calibrate() first.")
        
        return transform_points(points, self.H)


if __name__ == "__main__":
    # Test homography computation
    # Simulated calibration points in image
    src = np.array([
        [100, 50],   # Top
        [200, 150],  # Right
        [100, 250],  # Bottom
        [50, 150],   # Left
    ], dtype=np.float32)
    
    # Create calibrator
    calibrator = DartboardCalibrator()
    
    # Calibrate
    success = calibrator.calibrate(src)
    print(f"Calibration success: {success}")
    print(f"Homography matrix:\n{calibrator.H}")
    print(f"Center: {calibrator.center}")
    print(f"Radius: {calibrator.radius}")
    
    # Test transformation
    test_points = np.array([[100, 150], [125, 100]])
    transformed = calibrator.transform_to_dartboard(test_points)
    print(f"\nTest points: {test_points}")
    print(f"Transformed: {transformed}")
