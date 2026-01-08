"""
Base service class
"""
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class ServiceResult:
    """Service result wrapper"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def is_success(self) -> bool:
        """Check if result is successful"""
        return self.success
    
    def is_failure(self) -> bool:
        """Check if result is failure"""
        return not self.success


class ApplicationService:
    """Base service class"""
    
    @classmethod
    def call(cls, *args, **kwargs):
        """Call service instance"""
        instance = cls(*args, **kwargs)
        return instance.execute()
    
    def execute(self):
        """Execute service logic - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def success(self, result: Any = None) -> ServiceResult:
        """Return successful result"""
        return ServiceResult(success=True, result=result)
    
    def failure(self, error: str) -> ServiceResult:
        """Return failure result"""
        return ServiceResult(success=False, error=error)

