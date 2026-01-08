"""
Segmentation processor background task
"""
from sqlalchemy.orm import Session
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models.ct_scan import CtScan
from app.services.liver_segmentation_service import LiverSegmentationService


@celery_app.task(name="segmentation_processor")
def process_segmentation(ct_scan_id: int):
    """Process segmentation for CT scan"""
    db: Session = SessionLocal()
    
    try:
        # Load CT scan
        ct_scan = db.query(CtScan).filter(CtScan.id == ct_scan_id).first()
        if not ct_scan:
            raise ValueError(f"CT scan {ct_scan_id} not found")
        
        # Run segmentation service
        service = LiverSegmentationService(ct_scan, db=db)
        result = service.execute()
        
        if not result.is_success():
            raise ValueError(f"Segmentation failed: {result.error}")
        
        return {
            "success": True,
            "task_id": result.result.id,
            "ct_scan_id": ct_scan_id
        }
    finally:
        db.close()

