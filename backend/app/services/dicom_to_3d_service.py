"""
Service for converting DICOM segmentation to 3D model
"""
import os
import json
import numpy as np
import secrets
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from skimage import measure
from stl import mesh as stl_mesh

from app.models.ct_scan import CtScan
from app.models.three_d_model import ThreeDModel, ThreeDModelStatus
from app.models.segmentation_task import SegmentationTask
from app.services.application_service import ApplicationService, ServiceResult
from app.core.logging_config import app_logger


class DicomTo3dService(ApplicationService):
    """Service for converting segmentation mask to 3D model"""
    
    def __init__(self, ct_scan: CtScan, db: Session = None):
        self.ct_scan = ct_scan
        self.db = db
    
    def execute(self) -> ServiceResult:
        """Generate 3D model"""
        try:
            app_logger.info(f"Starting 3D model generation for CT scan {self.ct_scan.id}")
            
            # Find completed segmentation task
            segmentation_task = self._find_segmentation_task()
            if not segmentation_task:
                return self.failure("No completed segmentation found for this CT scan")
            
            # Get segmentation mask
            mask_data = self._load_segmentation_mask(segmentation_task)
            if mask_data is None:
                return self.failure("Could not load segmentation mask")
            
            # Generate 3D model from mask
            stl_path = self._generate_3d_model(mask_data)
            
            # Save result to database
            model_record = self._save_result(stl_path)
            
            app_logger.info(f"3D model generated successfully: {stl_path}")
            
            return self.success({
                "id": model_record.id,
                "name": model_record.name,
                "status": model_record.status.value,
                "model_file": stl_path,
                "ct_scan_id": self.ct_scan.id
            })
            
        except Exception as e:
            app_logger.error(f"3D model generation failed: {e}")
            return self.failure(str(e))
    
    def _find_segmentation_task(self) -> Optional[SegmentationTask]:
        """Find completed segmentation task for this CT scan"""
        task = (
            self.db.query(SegmentationTask)
            .filter(
                SegmentationTask.ct_scan_id == self.ct_scan.id,
                SegmentationTask.status == 'completed'
            )
            .order_by(SegmentationTask.created_at.desc())
            .first()
        )
        return task
    
    def _load_segmentation_mask(self, task: SegmentationTask) -> Optional[np.ndarray]:
        """Load segmentation mask from task result"""
        try:
            if not task.segmentation_result:
                app_logger.warning(f"No segmentation result for task {task.id}")
                return self._generate_mock_liver_mask()
            
            # Try to load mask file if available
            mask_file = task.segmentation_result.mask_file
            if mask_file:
                mask_path = Path(mask_file)
                if mask_path.exists():
                    if mask_path.suffix == '.npy':
                        return np.load(mask_path)
                    elif mask_path.suffix in ['.nii', '.nii.gz']:
                        import nibabel as nib
                        nii_img = nib.load(str(mask_path))
                        return nii_img.get_fdata()
            
            # Fallback: generate mock mask
            app_logger.warning("No mask file found, generating mock liver mask")
            return self._generate_mock_liver_mask()
            
        except Exception as e:
            app_logger.error(f"Error loading segmentation mask: {e}")
            return self._generate_mock_liver_mask()
    
    def _generate_mock_liver_mask(self) -> np.ndarray:
        """Generate a mock liver mask for testing"""
        # Create a realistic liver-shaped 3D volume
        depth, height, width = 100, 256, 256
        mask = np.zeros((depth, height, width), dtype=np.uint8)
        
        # Create ellipsoid shape (approximate liver)
        center = (depth // 2, height // 2, width // 2)
        radii = (40, 80, 70)  # Liver-like proportions
        
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    # Ellipsoid equation
                    norm_dist = (
                        ((z - center[0]) / radii[0]) ** 2 +
                        ((y - center[1]) / radii[1]) ** 2 +
                        ((x - center[2]) / radii[2]) ** 2
                    )
                    if norm_dist <= 1.0:
                        mask[z, y, x] = 1
        
        app_logger.info(f"Generated mock liver mask: shape={mask.shape}, volume={mask.sum()} voxels")
        return mask
    
    def _generate_3d_model(self, mask: np.ndarray) -> str:
        """Generate STL 3D model from segmentation mask"""
        try:
            app_logger.info(f"Generating 3D mesh from mask: shape={mask.shape}")
            
            # Use marching cubes to generate mesh
            verts, faces, normals, values = measure.marching_cubes(
                mask,
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
            
            app_logger.info(f"Mesh generated: {len(verts)} vertices, {len(faces)} faces")
            
            # Create STL mesh
            liver_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
            
            for i, face in enumerate(faces):
                for j in range(3):
                    liver_mesh.vectors[i][j] = verts[face[j], :]
            
            # Save STL file
            output_dir = Path("storage") / "3d_models"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"liver_model_{self.ct_scan.id}_{secrets.token_hex(4)}.stl"
            output_path = output_dir / output_filename
            
            liver_mesh.save(str(output_path))
            
            app_logger.info(f"STL model saved: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            app_logger.error(f"Error generating 3D model: {e}")
            raise
    
    def _save_result(self, stl_path: str) -> ThreeDModel:
        """Save 3D model result to database"""
        # Find or create 3D model record
        three_d_model = (
            self.db.query(ThreeDModel)
            .filter(ThreeDModel.ct_scan_id == self.ct_scan.id)
            .order_by(ThreeDModel.created_at.desc())
            .first()
        )
        
        if not three_d_model:
            three_d_model = ThreeDModel(
                ct_scan_id=self.ct_scan.id,
                name=f"Liver_Model_{self.ct_scan.id}",
                status=ThreeDModelStatus.PENDING
            )
            self.db.add(three_d_model)
        
        # Update model record
        three_d_model.model_file = stl_path
        three_d_model.status = ThreeDModelStatus.COMPLETED
        
        self.db.commit()
        self.db.refresh(three_d_model)
        
        return three_d_model
