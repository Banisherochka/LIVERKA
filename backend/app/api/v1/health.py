"""
Health check endpoint
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

