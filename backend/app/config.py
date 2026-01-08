"""
Application configuration
"""
import os
from typing import Optional, Any
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Liver Segmentation API"
    APP_VERSION: str = "1.0.0"
    
    # Эти поля должны быть объявлены для переменных из .env
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/liver_segmentation"
    REDIS_URL: str = "redis://localhost:6379/0"
    STORAGE_PATH: str = "./storage"
    MAX_UPLOAD_SIZE: int = 100000000  # 100MB
    CORS_ORIGINS: str = "http://localhost:4200"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Neural Network Service
    NEURAL_NETWORK_SERVICE_URL: Optional[str] = None
    
    # Валидаторы для преобразования типов
    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 't', 'yes', 'y')
        return bool(v)
    
    @field_validator('MAX_UPLOAD_SIZE', mode='before')
    @classmethod
    def parse_int(cls, v):
        if isinstance(v, str):
            return int(v)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False  # Делаем нечувствительным к регистру
        extra = "allow"  # Разрешаем дополнительные поля


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()