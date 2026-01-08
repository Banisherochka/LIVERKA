"""
Service for processing DICOM files
"""
import os
import json
import secrets
from datetime import date
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from app.models.ct_scan import CtScan, CtScanStatus
from app.services.application_service import ApplicationService, ServiceResult
from app.config import get_settings

settings = get_settings()


class DicomProcessingService(ApplicationService):
    """Service for processing DICOM files"""
    
    def __init__(self, file, patient_id: Optional[str] = None, db: Session = None):
        self.file = file
        self.patient_id = patient_id
        self.db = db
        self.error = None
        if not self.patient_id:
            self.patient_id = self._generate_anonymous_id()
    
    def execute(self) -> ServiceResult:
        """Process DICOM file"""
        if not self.file:
            return self.failure("No file provided")
        
        try:
            return self._process_dicom_file()
        except Exception as e:
            return self.failure(str(e))
    
    def _process_dicom_file(self) -> ServiceResult:
        """Main DICOM processing logic"""
        # Extract metadata
        metadata = self._extract_metadata()
        
        # Create CT scan record
        ct_scan = self._create_ct_scan(metadata)
        
        # Save DICOM file
        self._save_dicom_file(ct_scan)
        
        # Process slices
        self._process_slices(ct_scan)
        
        return self.success(ct_scan)
    
    def _extract_metadata(self) -> dict:
        """Extract metadata from DICOM file"""
        filename = getattr(self.file, 'filename', 'unknown.dcm')
        
        return {
            "patient_id": self._anonymize_patient_id(),
            "study_date": self._extract_study_date(),
            "modality": self._detect_modality(filename),
            "slice_count": self._estimate_slice_count(),
            "series_description": "CT Abdomen with Contrast",
            "institution_name": "Anonymous Hospital",
            "manufacturer": "Unknown"
        }
    
    def _create_ct_scan(self, metadata: dict) -> CtScan:
        """Create CT scan record"""
        ct_scan = CtScan(
            patient_id=metadata["patient_id"],
            study_date=metadata["study_date"],
            modality=metadata["modality"],
            slice_count=metadata["slice_count"],
            status=CtScanStatus.UPLOADED,
            dicom_series=json.dumps(metadata)
        )
        self.db.add(ct_scan)
        self.db.commit()
        self.db.refresh(ct_scan)
        return ct_scan
    
    def _save_dicom_file(self, ct_scan: CtScan):
        """Save DICOM file to storage"""
        from app.core.security import sanitize_filename
        from app.core.logging_config import app_logger
        
        # Use absolute path to prevent path traversal
        storage_base = Path(settings.STORAGE_PATH).resolve()
        storage_path = storage_base / "dicom_files"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        filename = self._sanitize_filename()
        # Use only filename, no path components
        safe_filename = sanitize_filename(filename)
        file_path = (storage_path / f"{ct_scan.id}_{safe_filename}").resolve()
        
        # Security: Ensure file is within storage directory
        try:
            file_path.relative_to(storage_base)
        except ValueError:
            app_logger.error(f"SECURITY: Path traversal attempt detected: {file_path}")
            raise ValueError("Invalid file path")
        
        # Save file
        try:
            if hasattr(self.file, 'read'):
                content = self.file.read()
                if isinstance(content, str):
                    content = content.encode('utf-8')
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                # If it's a path
                import shutil
                shutil.copy(self.file, file_path)
            
            # Update CT scan with file path (store relative path)
            ct_scan.dicom_file = str(file_path.relative_to(storage_base))
            self.db.commit()
            
            app_logger.info(f"DICOM file saved: {file_path}")
        except Exception as e:
            app_logger.error(f"Error saving DICOM file: {e}")
            raise
    
    def _process_slices(self, ct_scan: CtScan):
        """Process CT scan slices"""
        # TODO: Extract individual slices from DICOM series
        pass
    
    def _anonymize_patient_id(self) -> str:
        """Anonymize patient ID"""
        import re
        if re.match(r'^[A-Z0-9_-]+$', self.patient_id, re.IGNORECASE):
            return self.patient_id
        return self._generate_anonymous_id()
    
    def _generate_anonymous_id(self) -> str:
        """Generate anonymous patient ID"""
        return f"ANON_{secrets.token_hex(8).upper()}"
    
    def _extract_study_date(self) -> date:
        """Extract study date"""
        # TODO: Extract from DICOM metadata
        return date.today()
    
    def _detect_modality(self, filename: str) -> str:
        """Detect modality from filename"""
        filename_lower = filename.lower()
        if 'ct' in filename_lower:
            return 'CT'
        if 'mr' in filename_lower or 'mri' in filename_lower:
            return 'MR'
        return 'CT'
    
    def _estimate_slice_count(self) -> int:
        """Estimate slice count from file size"""
        if hasattr(self.file, 'size'):
            file_size = self.file.size
        elif hasattr(self.file, 'read'):
            # Read to get size
            pos = self.file.tell()
            self.file.seek(0, 2)  # Seek to end
            file_size = self.file.tell()
            self.file.seek(pos)  # Restore position
        else:
            file_size = os.path.getsize(self.file)
        
        # Rough estimate: 512x512 slice ≈ 0.5 MB
        estimated = int(file_size / (512 * 1024))
        return max(estimated, 1)
    
    def _sanitize_filename(self) -> str:
        """Sanitize filename"""
        from app.core.security import sanitize_filename
        
        original = getattr(self.file, 'filename', 'dicom_file.dcm')
        sanitized = sanitize_filename(original)
        
        # Add random suffix for uniqueness
        ext = Path(sanitized).suffix
        name = Path(sanitized).stem
        return f"{name}_{secrets.token_hex(4)}{ext}"

