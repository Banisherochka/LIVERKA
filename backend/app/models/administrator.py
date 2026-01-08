"""
Administrator model
"""
from sqlalchemy import Column, Integer, String, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel


class AdministratorRole(str, enum.Enum):
    """Administrator role enumeration"""
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Administrator(BaseModel):
    """Model for administrators"""
    __tablename__ = "administrators"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, nullable=True)
    hashed_password = Column(String, nullable=True)
    password_digest = Column(String, nullable=True)  # Legacy field
    role = Column(
        SQLEnum(AdministratorRole),
        nullable=False,
        default=AdministratorRole.ADMIN,
        index=True
    )
    first_login = Column(Boolean, default=True)
    
    # Relationships
    admin_oplogs = relationship(
        "AdminOplog",
        back_populates="administrator",
        cascade="all, delete-orphan"
    )
    
    def is_super_admin(self) -> bool:
        """Check if administrator is super admin"""
        return self.role == AdministratorRole.SUPER_ADMIN
    
    def is_admin(self) -> bool:
        """Check if administrator is admin"""
        return self.role == AdministratorRole.ADMIN
    
    def can_manage_administrators(self) -> bool:
        """Check if can manage administrators"""
        return self.is_super_admin()
    
    def can_delete_administrators(self) -> bool:
        """Check if can delete administrators"""
        return self.is_super_admin()
    
    def can_be_deleted_by(self, current_admin: "Administrator") -> bool:
        """Check if can be deleted by current admin"""
        if not current_admin.can_delete_administrators():
            return False
        # Super admin cannot delete themselves
        if self.id == current_admin.id:
            return False
        return True
    
    @property
    def role_name(self) -> str:
        """Get display role name"""
        if self.role == AdministratorRole.SUPER_ADMIN:
            return "Super Admin"
        elif self.role == AdministratorRole.ADMIN:
            return "Admin"
        return self.role.value.title()

