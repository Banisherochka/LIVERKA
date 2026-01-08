"""
Segmentation Result model
"""
from sqlalchemy import Column, Integer, String, Numeric, ForeignKey, JSON
from sqlalchemy.orm import relationship
from decimal import Decimal
from typing import Optional, Dict, Any

from app.models.base import BaseModel


class SegmentationResult(BaseModel):
    """Model for segmentation results"""
    __tablename__ = "segmentation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    segmentation_task_id = Column(
        Integer,
        ForeignKey("segmentation_tasks.id"),
        nullable=False,
        index=True
    )
    mask_file = Column(String, nullable=True)  # Path to mask file
    contours = Column(JSON, nullable=True)  # Contour data
    metrics = Column(JSON, nullable=True)  # Additional metrics
    volume_ml = Column(Numeric(10, 2), nullable=True)
    dice_coefficient = Column(Numeric(5, 4), nullable=True)
    iou_score = Column(Numeric(5, 4), nullable=True)
    
    # Relationships
    segmentation_task = relationship(
        "SegmentationTask",
        back_populates="segmentation_result"
    )
    
    @property
    def quality_grade(self) -> str:
        """Get quality grade based on Dice coefficient"""
        if not self.dice_coefficient:
            return "N/A"
        
        dice = float(self.dice_coefficient)
        if dice >= 0.90:
            return "Excellent"
        elif dice >= 0.80:
            return "Good"
        elif dice >= 0.70:
            return "Fair"
        else:
            return "Poor"
    
    def meets_clinical_standards(self) -> bool:
        """Check if result meets clinical standards"""
        if not self.dice_coefficient or not self.iou_score:
            return False
        return float(self.dice_coefficient) >= 0.90 and float(self.iou_score) >= 0.90
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of results"""
        return {
            "dice": round(float(self.dice_coefficient), 4) if self.dice_coefficient else None,
            "iou": round(float(self.iou_score), 4) if self.iou_score else None,
            "volume_ml": round(float(self.volume_ml), 2) if self.volume_ml else None,
            "quality": self.quality_grade,
            "clinical_grade": self.meets_clinical_standards()
        }

