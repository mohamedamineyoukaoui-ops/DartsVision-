"""
Complete IoU (CIoU) Loss for YOLO-style Keypoint Detection
Based on the official DeepDarts paper and YOLOv4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def bbox_iou(boxes1, boxes2, mode='iou'):
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        boxes1: (N, 4) tensor [x, y, w, h]
        boxes2: (M, 4) tensor [x, y, w, h]
        mode: 'iou', 'giou', 'diou', or 'ciou'
        
    Returns:
        iou: (N, M) IoU matrix
    """
    # Convert to x1y1x2y2 format
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
    
    b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
    b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
    b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
    b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1.T)
    inter_y1 = torch.max(b1_y1, b2_y1.T)
    inter_x2 = torch.min(b1_x2, b2_x2.T)
    inter_y2 = torch.min(b1_y2, b2_y2.T)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    iou = inter_area / (union + 1e-7)
    
    if mode == 'iou':
        return iou
    
    # GIoU
    enclose_x1 = torch.min(b1_x1, b2_x1.T)
    enclose_y1 = torch.min(b1_y1, b2_y1.T)
    enclose_x2 = torch.max(b1_x2, b2_x2.T)
    enclose_y2 = torch.max(b1_y2, b2_y2.T)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    giou = iou - (enclose_area - union) / (enclose_area + 1e-7)
    
    if mode == 'giou':
        return giou
    
    # DIoU / CIoU
    center1_x = boxes1[:, 0:1]
    center1_y = boxes1[:, 1:2]
    center2_x = boxes2[:, 0:1]
    center2_y = boxes2[:, 1:2]
    
    center_dist = (center1_x - center2_x.T) ** 2 + (center1_y - center2_y.T) ** 2
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    diou = iou - center_dist / (enclose_diag + 1e-7)
    
    if mode == 'diou':
        return diou
    
    # CIoU
    w1, h1 = boxes1[:, 2:3], boxes1[:, 3:4]
    w2, h2 = boxes2[:, 2:3], boxes2[:, 3:4]
    
    v = (4 / (np.pi ** 2)) * torch.pow(
        torch.atan(w2.T / (h2.T + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2
    )
    alpha = v / (1 - iou + v + 1e-7)
    
    ciou = diou - alpha * v
    
    return ciou


def ciou_loss(pred, target):
    """
    Compute CIoU loss between predicted and target boxes.
    
    Args:
        pred: (N, 4) predicted boxes [x, y, w, h]
        target: (N, 4) target boxes [x, y, w, h]
        
    Returns:
        loss: CIoU loss (scalar)
    """
    # CIoU between matched pairs
    b1_x1 = pred[:, 0] - pred[:, 2] / 2
    b1_y1 = pred[:, 1] - pred[:, 3] / 2
    b1_x2 = pred[:, 0] + pred[:, 2] / 2
    b1_y2 = pred[:, 1] + pred[:, 3] / 2
    
    b2_x1 = target[:, 0] - target[:, 2] / 2
    b2_y1 = target[:, 1] - target[:, 3] / 2
    b2_x2 = target[:, 0] + target[:, 2] / 2
    b2_y2 = target[:, 1] + target[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = pred[:, 2] * pred[:, 3]
    area2 = target[:, 2] * target[:, 3]
    union = area1 + area2 - inter_area
    
    iou = inter_area / (union + 1e-7)
    
    # Enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    
    # Center distance
    center_dist = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    diou = iou - center_dist / (enclose_diag + 1e-7)
    
    # Aspect ratio penalty
    v = (4 / (np.pi ** 2)) * torch.pow(
        torch.atan(target[:, 2] / (target[:, 3] + 1e-7)) - 
        torch.atan(pred[:, 2] / (pred[:, 3] + 1e-7)), 2
    )
    alpha = v / (1 - iou + v + 1e-7)
    
    ciou = diou - alpha * v
    
    return 1 - ciou.mean()


class YOLOv4Loss(nn.Module):
    """
    YOLOv4-style loss for keypoint detection.
    Includes CIoU loss for box regression, BCE for objectness and classification.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        num_anchors: int = 3,
        input_size: int = 800,
        bbox_size: float = 0.025,
        ignore_thresh: float = 0.5,
        coord_scale: float = 1.0,
        obj_scale: float = 1.0,
        noobj_scale: float = 0.5,
        cls_scale: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.bbox_size = bbox_size
        self.ignore_thresh = ignore_thresh
        
        self.coord_scale = coord_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.cls_scale = cls_scale
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets, anchors_list, strides):
        """
        Compute YOLO loss.
        
        Args:
            predictions: List of predictions at different scales
                        Each: (B, A*(5+C), H, W)
            targets: List of bbox targets per image in batch
                    Each: (N, 5) with [x, y, w, h, class]
            anchors_list: List of anchor tensors for each scale
            strides: List of stride values for each scale
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        total_coord_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_noobj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        
        for pred, anchors, stride in zip(predictions, anchors_list, strides):
            batch, channels, h, w = pred.shape
            
            # Reshape: (B, A*(5+C), H, W) -> (B, A, H, W, 5+C)
            pred = pred.view(batch, self.num_anchors, 5 + self.num_classes, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            
            # Decode predictions
            pred_xy = (torch.sigmoid(pred[..., :2]) + grid) / torch.tensor([w, h], device=device)
            pred_wh = torch.exp(pred[..., 2:4]) * anchors.view(1, self.num_anchors, 1, 1, 2) / self.input_size
            pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            # Build targets for this scale
            obj_mask = torch.zeros(batch, self.num_anchors, h, w, device=device)
            noobj_mask = torch.ones(batch, self.num_anchors, h, w, device=device)
            target_boxes = torch.zeros(batch, self.num_anchors, h, w, 4, device=device)
            target_cls = torch.zeros(batch, self.num_anchors, h, w, dtype=torch.long, device=device)
            
            for b in range(batch):
                if len(targets[b]) == 0:
                    continue
                
                gt_boxes = targets[b][:, :4].to(device)  # x, y, w, h
                gt_cls = targets[b][:, 4].long().to(device)
                
                # Find which anchor/grid cell each target belongs to
                for i in range(len(gt_boxes)):
                    gx = int(gt_boxes[i, 0] * w)
                    gy = int(gt_boxes[i, 1] * h)
                    
                    if gx < 0 or gx >= w or gy < 0 or gy >= h:
                        continue
                    
                    # Find best anchor (for keypoints, all anchors are similar)
                    best_anchor = 0  # Use first anchor for simplicity
                    
                    obj_mask[b, best_anchor, gy, gx] = 1
                    noobj_mask[b, best_anchor, gy, gx] = 0
                    target_boxes[b, best_anchor, gy, gx] = gt_boxes[i]
                    target_cls[b, best_anchor, gy, gx] = gt_cls[i]
            
            # Losses
            # Coordinate loss (CIoU)
            pos_mask = obj_mask.bool()
            if pos_mask.sum() > 0:
                pred_pos = pred_boxes[pos_mask]
                target_pos = target_boxes[pos_mask]
                coord_loss = ciou_loss(pred_pos, target_pos)
            else:
                coord_loss = torch.tensor(0.0, device=device)
            
            # Objectness loss
            obj_loss = self.bce(pred_obj[pos_mask], torch.ones_like(pred_obj[pos_mask]))
            noobj_loss = self.bce(pred_obj[noobj_mask.bool()], torch.zeros_like(pred_obj[noobj_mask.bool()]))
            
            obj_loss = obj_loss.mean() if obj_loss.numel() > 0 else torch.tensor(0.0, device=device)
            noobj_loss = noobj_loss.mean() if noobj_loss.numel() > 0 else torch.tensor(0.0, device=device)
            
            # Classification loss
            if pos_mask.sum() > 0:
                cls_loss = self.ce(pred_cls[pos_mask], target_cls[pos_mask])
                cls_loss = cls_loss.mean()
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            total_coord_loss += coord_loss * self.coord_scale
            total_obj_loss += obj_loss * self.obj_scale
            total_noobj_loss += noobj_loss * self.noobj_scale
            total_cls_loss += cls_loss * self.cls_scale
        
        total_loss = total_coord_loss + total_obj_loss + total_noobj_loss + total_cls_loss
        
        return total_loss, {
            'coord': total_coord_loss.item(),
            'obj': total_obj_loss.item(),
            'noobj': total_noobj_loss.item(),
            'cls': total_cls_loss.item(),
            'total': total_loss.item()
        }


if __name__ == "__main__":
    # Test the loss function
    loss_fn = YOLOv4Loss(num_classes=5)
    
    # Dummy predictions
    pred1 = torch.randn(2, 30, 50, 50)  # Scale 1: /16
    pred2 = torch.randn(2, 30, 25, 25)  # Scale 2: /32
    
    # Dummy targets
    targets = [
        torch.tensor([[0.5, 0.5, 0.025, 0.025, 1]]),  # Batch 0
        torch.tensor([[0.3, 0.3, 0.025, 0.025, 0], [0.7, 0.7, 0.025, 0.025, 2]]),  # Batch 1
    ]
    
    anchors = [
        torch.tensor([[20, 20], [25, 25], [30, 30]], dtype=torch.float32),
        torch.tensor([[40, 40], [50, 50], [60, 60]], dtype=torch.float32),
    ]
    strides = [16, 32]
    
    loss, loss_dict = loss_fn([pred1, pred2], targets, anchors, strides)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
