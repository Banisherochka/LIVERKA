"""
CT Scan model
"""
from sqlalchemy import Column, String, Text, Date, Integer, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel


class CtScanStatus(str, enum.Enum):
    """CT Scan status enumeration"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CtScan(BaseModel):
    """Model for storing CT scan information"""
    __tablename__ = "ct_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    dicom_series = Column(Text)  # JSON metadata
    dicom_file = Column(String, nullable=True)  # Path to DICOM file
    study_date = Column(Date)
    modality = Column(String, default="CT", nullable=False)
    slice_count = Column(Integer, default=0)
    status = Column(
        SQLEnum(CtScanStatus),
        default=CtScanStatus.UPLOADED,
        nullable=False
    )
    
    # Relationships
    segmentation_tasks = relationship(
        "SegmentationTask",
        back_populates="ct_scan",
        cascade="all, delete-orphan"
    )
    three_d_models = relationship(
        "ThreeDModel",
        back_populates="ct_scan",
        cascade="all, delete-orphan"
    )
    
    def is_processed(self) -> bool:
        """Check if CT scan is processed"""
        return self.status == CtScanStatus.COMPLETED
    
    def is_processing(self) -> bool:
        """Check if CT scan is being processed"""
        return self.status == CtScanStatus.PROCESSING
    
    def is_failed(self) -> bool:
        """Check if CT scan processing failed"""
        return self.status == CtScanStatus.FAILED

