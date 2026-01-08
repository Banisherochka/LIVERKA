"""
Segmentation Task model
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum
from datetime import datetime
from typing import Optional

from app.models.base import BaseModel


class SegmentationTaskStatus(str, enum.Enum):
    """Segmentation task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SegmentationTask(BaseModel):
    """Model for segmentation tasks"""
    __tablename__ = "segmentation_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    ct_scan_id = Column(Integer, ForeignKey("ct_scans.id"), nullable=False, index=True)
    status = Column(
        SQLEnum(SegmentationTaskStatus),
        default=SegmentationTaskStatus.PENDING,
        nullable=False
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    inference_time_ms = Column(Integer, default=0)
    
    # Relationships
    ct_scan = relationship("CtScan", back_populates="segmentation_tasks")
    segmentation_result = relationship(
        "SegmentationResult",
        back_populates="segmentation_task",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    def is_pending(self) -> bool:
        """Check if task is pending"""
        return self.status == SegmentationTaskStatus.PENDING
    
    def is_processing(self) -> bool:
        """Check if task is processing"""
        return self.status == SegmentationTaskStatus.PROCESSING
    
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.status == SegmentationTaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed"""
        return self.status == SegmentationTaskStatus.FAILED
    
    def mark_as_processing(self):
        """Mark task as processing"""
        self.status = SegmentationTaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
    
    def mark_as_completed(self, inference_time_ms: Optional[int] = None):
        """Mark task as completed"""
        self.status = SegmentationTaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if inference_time_ms is not None:
            self.inference_time_ms = inference_time_ms
        elif self.started_at:
            duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
            self.inference_time_ms = duration_ms
    
    def mark_as_failed(self, error_msg: str):
        """Mark task as failed"""
        self.status = SegmentationTaskStatus.FAILED
        self.error_message = error_msg
        self.completed_at = datetime.utcnow()
    
    def duration_ms(self) -> Optional[int]:
        """Calculate task duration in milliseconds"""
        if not self.completed_at or not self.started_at:
            return None
        return int((self.completed_at - self.started_at).total_seconds() * 1000)

