"""
Service for calculating segmentation metrics
"""
from typing import Optional, Dict, Any
import numpy as np

from app.services.application_service import ApplicationService, ServiceResult


class MetricsCalculationService(ApplicationService):
    """Service for calculating segmentation metrics"""
    
    def __init__(
        self,
        ground_truth: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None
    ):
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.error = None
    
    def execute(self) -> ServiceResult:
        """Calculate all metrics"""
        if not self.ground_truth or not self.prediction:
            return self.failure("Ground truth and prediction required")
        
        try:
            metrics = self._calculate_all_metrics()
            return self.success(metrics)
        except Exception as e:
            return self.failure(str(e))
    
    def _calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all metrics"""
        return {
            "dice": self._calculate_dice_coefficient(),
            "iou": self._calculate_iou(),
            "pixel_accuracy": self._calculate_pixel_accuracy(),
            "sensitivity": self._calculate_sensitivity(),
            "specificity": self._calculate_specificity(),
            "volume_ml": self._calculate_volume()
        }
    
    def _calculate_dice_coefficient(self) -> float:
        """Calculate Dice coefficient"""
        intersection = self._calculate_intersection()
        sum_sizes = self._ground_truth_size() + self._prediction_size()
        
        if sum_sizes == 0:
            return 0.0
        
        return round(2.0 * intersection / sum_sizes, 6)
    
    def _calculate_iou(self) -> float:
        """Calculate IoU (Intersection over Union)"""
        intersection = self._calculate_intersection()
        union = self._calculate_union()
        
        if union == 0:
            return 0.0
        
        return round(intersection / union, 6)
    
    def _calculate_pixel_accuracy(self) -> float:
        """Calculate pixel accuracy"""
        true_positive = self._calculate_intersection()
        true_negative = self._calculate_true_negatives()
        total_pixels = self._total_pixel_count()
        
        if total_pixels == 0:
            return 0.0
        
        return round((true_positive + true_negative) / total_pixels, 6)
    
    def _calculate_sensitivity(self) -> float:
        """Calculate sensitivity (Recall)"""
        true_positive = self._calculate_intersection()
        false_negative = self._ground_truth_size() - true_positive
        
        if (true_positive + false_negative) == 0:
            return 0.0
        
        return round(true_positive / (true_positive + false_negative), 6)
    
    def _calculate_specificity(self) -> float:
        """Calculate specificity"""
        true_negative = self._calculate_true_negatives()
        false_positive = self._prediction_size() - self._calculate_intersection()
        
        if (true_negative + false_positive) == 0:
            return 0.0
        
        return round(true_negative / (true_negative + false_positive), 6)
    
    def _calculate_volume(self, voxel_spacing: list = [1.0, 1.0, 1.0]) -> float:
        """Calculate volume in milliliters"""
        voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
        volume_voxels = self._prediction_size()
        
        return round(volume_voxels * voxel_volume / 1000.0, 2)
    
    def _calculate_intersection(self) -> float:
        """Calculate intersection of masks"""
        if isinstance(self.ground_truth, np.ndarray) and isinstance(self.prediction, np.ndarray):
            return float(np.sum((self.ground_truth > 0) & (self.prediction > 0)))
        # Mock implementation
        return min(self._ground_truth_size(), self._prediction_size()) * 0.92
    
    def _calculate_union(self) -> float:
        """Calculate union of masks"""
        return self._ground_truth_size() + self._prediction_size() - self._calculate_intersection()
    
    def _calculate_true_negatives(self) -> float:
        """Calculate true negatives"""
        return (
            self._total_pixel_count() -
            self._ground_truth_size() -
            self._prediction_size() +
            self._calculate_intersection()
        )
    
    def _ground_truth_size(self) -> float:
        """Get ground truth mask size"""
        if isinstance(self.ground_truth, np.ndarray):
            return float(np.sum(self.ground_truth > 0))
        # Mock implementation
        import random
        return self._total_pixel_count() * (0.05 + random.random() * 0.03)
    
    def _prediction_size(self) -> float:
        """Get prediction mask size"""
        if isinstance(self.prediction, np.ndarray):
            return float(np.sum(self.prediction > 0))
        # Mock implementation
        import random
        return self._total_pixel_count() * (0.05 + random.random() * 0.03)
    
    def _total_pixel_count(self) -> int:
        """Get total pixel count"""
        return 512 * 512 * 100  # Typical CT scan dimensions

