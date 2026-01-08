"""
CT Scans API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.api.dependencies import get_database
from app.models.ct_scan import CtScan
from app.models.three_d_model import ThreeDModel
from app.services.dicom_to_3d_service import DicomTo3dService

router = APIRouter()


@router.get("/ct_scans")
async def list_ct_scans(db: Session = Depends(get_database)):
    """List all CT scans"""
    scans = db.query(CtScan).order_by(CtScan.created_at.desc()).all()
    return {
        "success": True,
        "data": [_ct_scan_summary(scan) for scan in scans]
    }


@router.get("/ct_scans/{scan_id}")
async def get_ct_scan(scan_id: int, db: Session = Depends(get_database)):
    """Get CT scan details"""
    scan = db.query(CtScan).filter(CtScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="CT scan not found")
    
    return {
        "success": True,
        "data": _ct_scan_detail(scan)
    }


@router.get("/ct_scans/{scan_id}/three_d_models")
async def list_three_d_models(scan_id: int, db: Session = Depends(get_database)):
    """List 3D models for CT scan"""
    scan = db.query(CtScan).filter(CtScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="CT scan not found")
    
    models = db.query(ThreeDModel).filter(ThreeDModel.ct_scan_id == scan_id).all()
    
    return {
        "success": True,
        "data": [_three_d_model_summary(model) for model in models]
    }


@router.post("/ct_scans/{scan_id}/generate_3d")
async def generate_3d(scan_id: int, db: Session = Depends(get_database)):
    """Generate 3D model from CT scan"""
    scan = db.query(CtScan).filter(CtScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="CT scan not found")
    
    service = DicomTo3dService(scan, db=db)
    result = service.execute()
    
    if not result.is_success():
        raise HTTPException(status_code=422, detail=result.error)
    
    return {
        "success": True,
        "data": result.result
    }


def _ct_scan_summary(scan: CtScan) -> dict:
    """Get CT scan summary"""
    return {
        "id": scan.id,
        "patient_id": scan.patient_id,
        "modality": scan.modality,
        "status": scan.status.value,
        "slice_count": scan.slice_count,
        "created_at": scan.created_at.isoformat()
    }


def _ct_scan_detail(scan: CtScan) -> dict:
    """Get CT scan detail"""
    return {
        "id": scan.id,
        "patient_id": scan.patient_id,
        "study_date": scan.study_date.isoformat() if scan.study_date else None,
        "modality": scan.modality,
        "slice_count": scan.slice_count,
        "status": scan.status.value,
        "created_at": scan.created_at.isoformat(),
        "updated_at": scan.updated_at.isoformat()
    }


def _three_d_model_summary(model: ThreeDModel) -> dict:
    """Get 3D model summary"""
    return {
        "id": model.id,
        "name": model.name,
        "status": model.status.value,
        "created_at": model.created_at.isoformat()
    }

