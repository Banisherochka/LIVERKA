"""
API v1 routes
"""
from app.api.v1 import health, segmentations, ct_scans

__all__ = ["health", "segmentations", "ct_scans"]
