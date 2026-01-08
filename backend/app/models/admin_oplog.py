"""
Admin Operation Log model
"""
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class AdminOplog(BaseModel):
    """Model for admin operation logs"""
    __tablename__ = "admin_oplogs"
    
    id = Column(Integer, primary_key=True, index=True)
    administrator_id = Column(
        Integer,
        ForeignKey("administrators.id"),
        nullable=False,
        index=True
    )
    action = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=True)
    resource_id = Column(Integer, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    details = Column(Text, nullable=True)
    
    # Relationships
    administrator = relationship("Administrator", back_populates="admin_oplogs")
    
    # Indexes
    __table_args__ = (
        {"mysql_engine": "InnoDB"},
    )

