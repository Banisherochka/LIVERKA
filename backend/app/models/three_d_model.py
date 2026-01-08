"""
3D Model model
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel


class ThreeDModelStatus(str, enum.Enum):
    """3D Model status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ThreeDModel(BaseModel):
    """Model for 3D models"""
    __tablename__ = "three_d_models"
    
    id = Column(Integer, primary_key=True, index=True)
    ct_scan_id = Column(Integer, ForeignKey("ct_scans.id"), nullable=False)
    name = Column(String, nullable=False)
    model_file = Column(String, nullable=True)  # Path to model file
    status = Column(
        SQLEnum(ThreeDModelStatus),
        default=ThreeDModelStatus.PENDING,
        nullable=False
    )
    
    # Relationships
    ct_scan = relationship("CtScan", back_populates="three_d_models")

