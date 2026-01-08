"""
Main API router
"""
from fastapi import APIRouter

from app.api.v1 import health, segmentations, ct_scans, auth

api_router = APIRouter()

# Include v1 routes
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(auth.router, prefix="/v1", tags=["authentication"])
api_router.include_router(segmentations.router, prefix="/v1", tags=["segmentations"])
api_router.include_router(ct_scans.router, prefix="/v1", tags=["ct_scans"])

