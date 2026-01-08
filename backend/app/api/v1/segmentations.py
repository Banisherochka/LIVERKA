"""
Segmentation API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.api.dependencies import get_database
from app.models.ct_scan import CtScan
from app.models.segmentation_task import SegmentationTask
from app.services.dicom_processing_service import DicomProcessingService
from app.services.liver_segmentation_service import LiverSegmentationService

router = APIRouter()


@router.post("/segmentation/upload")
def upload_segmentation(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    db: Session = Depends(get_database)
):
    """Upload DICOM file and start segmentation"""
    from app.core.security import (
        sanitize_filename,
        validate_file_extension,
        validate_file_size
    )
    from app.core.logging_config import app_logger
    from app.config import get_settings
    
    settings = get_settings()
    
    if not file:
        raise HTTPException(status_code=400, detail="File parameter required")
    
    # Security: Validate filename
    original_filename = file.filename or "unknown.dcm"
    safe_filename = sanitize_filename(original_filename)
    
    if not validate_file_extension(safe_filename):
        app_logger.warning(f"SECURITY: Invalid file extension: {original_filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only DICOM files (.dcm) are allowed."
        )
    
    # Security: Check file size
    file_size = 0
    if hasattr(file, 'size'):
        file_size = file.size
    else:
        # Read content to check size
        content = file.read()
        file_size = len(content)
        # Reset file pointer
        file.seek(0)
    
    if not validate_file_size(file_size):
        app_logger.warning(f"SECURITY: File too large: {file_size} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB"
        )
    
    app_logger.info(f"Processing DICOM upload: {safe_filename} ({file_size} bytes)")
    
    try:
        # Process DICOM file
        dicom_service = DicomProcessingService(file=file, patient_id=patient_id, db=db)
        dicom_result = dicom_service.execute()
        
        if not dicom_result.is_success():
            app_logger.error(f"DICOM processing failed: {dicom_result.error}")
            raise HTTPException(
                status_code=422,
                detail=dicom_result.error
            )
        
        ct_scan = dicom_result.result
        
        # Start segmentation
        segmentation_service = LiverSegmentationService(ct_scan, db=db)
        segmentation_result = segmentation_service.execute()
        
        if not segmentation_result.is_success():
            app_logger.error(f"Segmentation failed: {segmentation_result.error}")
            raise HTTPException(
                status_code=422,
                detail=segmentation_result.error
            )
        
        task = segmentation_result.result
        
        app_logger.info(f"Segmentation task created: {task.id} for CT scan {ct_scan.id}")
        
        return {
            "success": True,
            "data": {
                "task_id": task.id,
                "ct_scan_id": ct_scan.id,
                "status": str(task.status.value),
                "message": "Segmentation task created successfully"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error in upload_segmentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during file processing"
        )


@router.post("/segmentations")
async def create_segmentation(
    ct_scan_id: int,
    db: Session = Depends(get_database)
):
    """Create segmentation task for existing CT scan"""
    ct_scan = db.query(CtScan).filter(CtScan.id == ct_scan_id).first()
    if not ct_scan:
        raise HTTPException(status_code=404, detail="CT scan not found")
    
    # Start segmentation
    segmentation_service = LiverSegmentationService(ct_scan, db=db)
    result = segmentation_service.execute()
    
    if not result.is_success():
        raise HTTPException(
            status_code=422,
            detail=result.error
        )
    
    task = result.result
    
    return {
        "success": True,
        "data": {
            "task_id": task.id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat()
        }
    }


@router.get("/segmentations")
async def list_segmentations(
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_database)
):
    """List all segmentation tasks"""
    tasks = (
        db.query(SegmentationTask)
        .order_by(SegmentationTask.created_at.desc())
        .limit(limit)
        .all()
    )
    
    return {
        "success": True,
        "data": {
            "tasks": [_task_summary(task) for task in tasks]
        }
    }


@router.get("/segmentations/{task_id}")
async def get_segmentation(
    task_id: int,
    db: Session = Depends(get_database)
):
    """Get segmentation task details"""
    task = (
        db.query(SegmentationTask)
        .filter(SegmentationTask.id == task_id)
        .first()
    )
    
    if not task:
        raise HTTPException(status_code=404, detail="Segmentation task not found")
    
    return {
        "success": True,
        "data": _task_detail(task)
    }


@router.get("/segmentations/{task_id}/result")
async def get_segmentation_result(
    task_id: int,
    db: Session = Depends(get_database)
):
    """Get segmentation result with metrics"""
    task = (
        db.query(SegmentationTask)
        .filter(SegmentationTask.id == task_id)
        .first()
    )
    
    if not task:
        raise HTTPException(status_code=404, detail="Segmentation task not found")
    
    if not task.is_completed():
        raise HTTPException(
            status_code=422,
            detail="Segmentation not completed"
        )
    
    result = task.segmentation_result
    if not result:
        raise HTTPException(status_code=404, detail="Segmentation result not found")
    
    return {
        "success": True,
        "data": {
            "task_id": task.id,
            "status": task.status.value,
            "inference_time_ms": task.inference_time_ms,
            "mask_file": result.mask_file,
            "contours": result.contours,
            "metrics": {
                "dice": float(result.dice_coefficient) if result.dice_coefficient else None,
                "iou": float(result.iou_score) if result.iou_score else None,
                "volume_ml": float(result.volume_ml) if result.volume_ml else None,
                "quality_grade": result.quality_grade,
                "meets_clinical_standards": result.meets_clinical_standards()
            },
            "summary": result.summary()
        }
    }


@router.get("/segmentations/{task_id}/download_mask")
async def download_mask(
    task_id: int,
    db: Session = Depends(get_database)
):
    """Download segmentation mask file"""
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    task = (
        db.query(SegmentationTask)
        .filter(SegmentationTask.id == task_id)
        .first()
    )
    
    if not task or not task.is_completed():
        raise HTTPException(
            status_code=422,
            detail="Segmentation not completed"
        )
    
    result = task.segmentation_result
    if not result or not result.mask_file:
        raise HTTPException(status_code=404, detail="Mask file not available")
    
    mask_path = Path(result.mask_file)
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask file not found")
    
    return FileResponse(
        mask_path,
        media_type="application/octet-stream",
        filename=mask_path.name
    )


def _task_summary(task: SegmentationTask) -> dict:
    """Get task summary"""
    return {
        "id": task.id,
        "ct_scan_id": task.ct_scan_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "inference_time_ms": task.inference_time_ms,
        "has_result": task.segmentation_result is not None
    }


def _task_detail(task: SegmentationTask) -> dict:
    """Get task detail"""
    return {
        "id": task.id,
        "ct_scan": {
            "id": task.ct_scan.id,
            "patient_id": task.ct_scan.patient_id,
            "study_date": task.ct_scan.study_date.isoformat() if task.ct_scan.study_date else None,
            "modality": task.ct_scan.modality,
            "slice_count": task.ct_scan.slice_count
        },
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "inference_time_ms": task.inference_time_ms,
        "error_message": task.error_message,
        "result": _result_summary(task.segmentation_result) if task.segmentation_result else None
    }


def _result_summary(result) -> dict:
    """Get result summary"""
    return {
        "dice_coefficient": float(result.dice_coefficient) if result.dice_coefficient else None,
        "iou_score": float(result.iou_score) if result.iou_score else None,
        "volume_ml": float(result.volume_ml) if result.volume_ml else None,
        "quality_grade": result.quality_grade,
        "meets_clinical_standards": result.meets_clinical_standards()
    }

