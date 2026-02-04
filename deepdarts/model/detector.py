"""
DeepDarts Detector: Modeling Keypoints as Objects
Based on the paper arXiv:2105.09880
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import YOLOv4TinyBackbone, ConvBNLeaky, SPPBlock


class DetectionHead(nn.Module):
    """
    YOLO detection head for keypoint-as-object detection.
    Predicts bounding boxes for keypoints.
    """
    
    def __init__(self, in_channels, num_classes=5, num_anchors=3):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of keypoint classes (4 calibration + 1 dart)
            num_anchors: Number of anchor boxes per cell
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Output: (tx, ty, tw, th, objectness, class_probs...)
        self.out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            ConvBNLeaky(in_channels, in_channels * 2, 3, padding=1),
            nn.Conv2d(in_channels * 2, self.out_channels, 1)
        )
    
    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale detection"""
    
    def __init__(self):
        super().__init__()
        
        # SPP for the largest feature map
        self.spp = SPPBlock(512, 256)
        
        # Upsampling path
        self.conv_up1 = ConvBNLeaky(256, 128, 1)
        self.conv_up2 = ConvBNLeaky(128, 64, 1)
        
        # Lateral connections
        self.lateral1 = ConvBNLeaky(256, 128, 1)
        self.lateral2 = ConvBNLeaky(128, 64, 1)
        
        # Merge convolutions
        self.merge1 = nn.Sequential(
            ConvBNLeaky(256, 128, 1),
            ConvBNLeaky(128, 256, 3, padding=1),
            ConvBNLeaky(256, 128, 1)
        )
        self.merge2 = nn.Sequential(
            ConvBNLeaky(128, 64, 1),
            ConvBNLeaky(64, 128, 3, padding=1),
            ConvBNLeaky(128, 64, 1)
        )
    
    def forward(self, feat_small, feat_medium, feat_large):
        """
        Forward pass through FPN.
        
        Args:
            feat_small: Features at /8 scale (128 channels)
            feat_medium: Features at /16 scale (256 channels)
            feat_large: Features at /32 scale (512 channels)
            
        Returns:
            Multi-scale features for detection heads
        """
        # Process largest scale
        p5 = self.spp(feat_large)
        
        # Upsample and merge with medium scale
        p5_up = F.interpolate(self.conv_up1(p5), size=feat_medium.shape[2:], mode='nearest')
        p4 = self.lateral1(feat_medium)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.merge1(p4)
        
        # Upsample and merge with small scale
        p4_up = F.interpolate(self.conv_up2(p4), size=feat_small.shape[2:], mode='nearest')
        p3 = self.lateral2(feat_small)
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.merge2(p3)
        
        return p3, p4, p5


class DeepDartsDetector(nn.Module):
    """
    DeepDarts: Complete keypoint-as-object detector.
    
    Detects:
    - 4 calibration points (classes 0-3) at dartboard edges
    - D dart positions (class 4) where dart tips land
    """
    
    # Dartboard calibration points (at intersections of sections)
    CALIBRATION_CLASSES = ['cal_5_20', 'cal_13_6', 'cal_17_3', 'cal_8_11']
    DART_CLASS = 'dart'
    
    def __init__(self, num_classes=5, input_size=416):
        """
        Args:
            num_classes: 5 = 4 calibration points + 1 dart class
            input_size: Expected input image size (square)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone
        self.backbone = YOLOv4TinyBackbone()
        
        # Feature Pyramid Network
        self.fpn = FPN()
        
        # Detection heads at different scales
        self.head_small = DetectionHead(64, num_classes)   # /8 - small objects
        self.head_medium = DetectionHead(128, num_classes)  # /16 - medium objects
        self.head_large = DetectionHead(256, num_classes)   # /32 - large objects
        
        # Anchor boxes (relative to grid cell, for keypoint bboxes ~2.5% of image)
        # These are tuned for keypoint detection
        self.register_buffer('anchors_small', torch.tensor([
            [10, 10], [15, 15], [20, 20]
        ], dtype=torch.float32))
        self.register_buffer('anchors_medium', torch.tensor([
            [25, 25], [35, 35], [45, 45]
        ], dtype=torch.float32))
        self.register_buffer('anchors_large', torch.tensor([
            [60, 60], [80, 80], [100, 100]
        ], dtype=torch.float32))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            predictions: List of detection outputs at each scale
        """
        # Extract multi-scale features
        feat_small, feat_medium, feat_large = self.backbone(x)
        
        # FPN processing
        p3, p4, p5 = self.fpn(feat_small, feat_medium, feat_large)
        
        # Detection heads
        out_small = self.head_small(p3)
        out_medium = self.head_medium(p4)
        out_large = self.head_large(p5)
        
        return [out_small, out_medium, out_large]
    
    def decode_predictions(self, predictions, conf_threshold=0.25, iou_threshold=0.3):
        """
        Decode raw predictions to keypoint coordinates.
        
        Args:
            predictions: List of outputs from forward()
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            
        Returns:
            calibration_points: Tensor of shape (4, 2) for calibration point coordinates
            dart_points: Tensor of shape (D, 2) for dart coordinates
            confidences: Tensor of shape (D,) for dart confidences
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        anchors_list = [self.anchors_small, self.anchors_medium, self.anchors_large]
        strides = [8, 16, 32]
        
        for pred, anchors, stride in zip(predictions, anchors_list, strides):
            batch, channels, h, w = pred.shape
            num_anchors = anchors.shape[0]
            
            # Reshape: (B, A*(5+C), H, W) -> (B, A, 5+C, H, W)
            pred = pred.view(batch, num_anchors, 5 + self.num_classes, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, H, W, 5+C)
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            
            # Decode boxes
            xy = (torch.sigmoid(pred[..., :2]) + grid) * stride
            wh = torch.exp(pred[..., 2:4]) * anchors.view(1, num_anchors, 1, 1, 2)
            objectness = torch.sigmoid(pred[..., 4:5])
            class_probs = torch.sigmoid(pred[..., 5:])
            
            # Combine objectness and class probability
            scores = objectness * class_probs
            
            # Get best class
            max_scores, class_ids = scores.max(dim=-1)
            
            # Flatten for NMS
            for b in range(batch_size):
                mask = max_scores[b].view(-1) > conf_threshold
                if mask.sum() > 0:
                    boxes_b = torch.cat([
                        xy[b].view(-1, 2)[mask] - wh[b].view(-1, 2)[mask] / 2,
                        xy[b].view(-1, 2)[mask] + wh[b].view(-1, 2)[mask] / 2
                    ], dim=-1)
                    scores_b = max_scores[b].view(-1)[mask]
                    classes_b = class_ids[b].view(-1)[mask]
                    
                    all_boxes.append(boxes_b)
                    all_scores.append(scores_b)
                    all_classes.append(classes_b)
        
        if len(all_boxes) == 0:
            return (
                torch.zeros(4, 2, device=device),
                torch.zeros(0, 2, device=device),
                torch.zeros(0, device=device)
            )
        
        # Concatenate all detections
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        classes = torch.cat(all_classes, dim=0)
        
        # Apply NMS per class
        calibration_points = torch.zeros(4, 2, device=device)
        dart_points = []
        dart_confidences = []
        
        for cls in range(self.num_classes):
            cls_mask = classes == cls
            if cls_mask.sum() == 0:
                continue
            
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # NMS
            keep = self._nms(cls_boxes, cls_scores, iou_threshold)
            
            if cls < 4:  # Calibration point
                if len(keep) > 0:
                    # Take highest confidence calibration point
                    best_idx = keep[0]
                    center = (cls_boxes[best_idx, :2] + cls_boxes[best_idx, 2:]) / 2
                    calibration_points[cls] = center
            else:  # Dart
                for idx in keep:
                    center = (cls_boxes[idx, :2] + cls_boxes[idx, 2:]) / 2
                    dart_points.append(center)
                    dart_confidences.append(cls_scores[idx])
        
        if dart_points:
            dart_points = torch.stack(dart_points, dim=0)
            dart_confidences = torch.stack(dart_confidences, dim=0)
        else:
            dart_points = torch.zeros(0, 2, device=device)
            dart_confidences = torch.zeros(0, device=device)
        
        return calibration_points, dart_points, dart_confidences
    
    def _nms(self, boxes, scores, iou_threshold):
        """Simple NMS implementation"""
        if boxes.shape[0] == 0:
            return []
        
        # Sort by score
        _, order = scores.sort(descending=True)
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            # Compute IoU
            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area_j - inter + 1e-6)
            
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return keep


if __name__ == "__main__":
    # Test the detector
    model = DeepDartsDetector(num_classes=5, input_size=416)
    x = torch.randn(1, 3, 416, 416)
    
    outputs = model(x)
    print("Output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")
    
    # Test decoding
    cal_points, dart_points, dart_conf = model.decode_predictions(outputs)
    print(f"\nCalibration points: {cal_points.shape}")
    print(f"Dart points: {dart_points.shape}")
    print(f"Dart confidences: {dart_conf.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
