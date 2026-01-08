"""
Service for converting DICOM to 3D model
"""
import os
import json
import subprocess
import secrets
import shutil
from pathlib import Path
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.models.ct_scan import CtScan
from app.models.three_d_model import ThreeDModel, ThreeDModelStatus
from app.services.application_service import ApplicationService, ServiceResult


class DicomTo3dService(ApplicationService):
    """Service for converting DICOM to 3D model"""
    
    def __init__(self, ct_scan: CtScan, db: Session = None):
        self.ct_scan = ct_scan
        self.db = db
    
    def execute(self) -> ServiceResult:
        """Generate 3D model"""
        try:
            # Collect DICOM files
            dicom_files = self._collect_dicom_files()
            
            # Generate 3D model via Python script
            result = self._execute_python_script(dicom_files)
            
            # Save result
            self._save_result(result)
            
            return self.success(result)
        except Exception as e:
            return self.failure(str(e))
    
    def _collect_dicom_files(self) -> list:
        """Collect DICOM files"""
        # Assuming DICOM files are stored in storage
        storage_path = Path("storage") / "dicom_files" / str(self.ct_scan.id)
        if storage_path.exists():
            return list(storage_path.glob("*.dcm"))
        return []
    
    def _execute_python_script(self, dicom_files: list) -> Dict[str, Any]:
        """Execute Python script for 3D generation"""
        script_path = Path("scripts") / "generate_3d.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        temp_dir = Path("tmp") / "dicom_processing" / secrets.token_hex(8)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy files to temp directory
            for index, file in enumerate(dicom_files):
                shutil.copy(file, temp_dir / f"slice_{index:04d}.dcm")
            
            # Run Python script
            result = subprocess.run(
                ["python3", str(script_path), str(temp_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            
            return json.loads(result.stdout)
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _save_result(self, result: Dict[str, Any]):
        """Save 3D model result"""
        stl_path = result.get("stl_path")
        if stl_path and os.path.exists(stl_path):
            # Find or create 3D model
            three_d_model = self.db.query(ThreeDModel).filter(
                ThreeDModel.ct_scan_id == self.ct_scan.id
            ).order_by(ThreeDModel.created_at.desc()).first()
            
            if not three_d_model:
                three_d_model = ThreeDModel(
                    ct_scan_id=self.ct_scan.id,
                    name=f"Model_{self.ct_scan.id}",
                    status=ThreeDModelStatus.PENDING
                )
                self.db.add(three_d_model)
            
            # Save file path
            three_d_model.model_file = stl_path
            three_d_model.status = ThreeDModelStatus.COMPLETED
            self.db.commit()

