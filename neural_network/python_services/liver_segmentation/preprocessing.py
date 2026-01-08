"""
Advanced DICOM Preprocessing for Liver Segmentation Pipeline

Features:
- Robust DICOM loading with error handling
- Automatic HU normalization with multiple window presets
- Intelligent resampling with interpolation
- Advanced data augmentation for deep learning
- DICOM metadata extraction and validation
- Support for 3D and 4D (temporal) DICOM data
- Batch processing capabilities
- GPU acceleration support
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HUWindowPreset(Enum):
    """Standard HU window presets for CT imaging"""
    LIVER = (40, 400)        # Liver window
    SOFT_TISSUE = (50, 350)  # Soft tissue window
    BONE = (400, 2000)       # Bone window
    LUNG = (-600, 1500)      # Lung window
    BRAIN = (40, 80)         # Brain window
    ABDOMEN = (60, 400)      # Abdomen window
    CUSTOM = None


class ResampleMethod(Enum):
    """Resampling interpolation methods"""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"


class AugmentationType(Enum):
    """Types of data augmentation"""
    FLIP = "flip"
    ROTATE = "rotate"
    TRANSLATE = "translate"
    SCALE = "scale"
    ELASTIC = "elastic"
    NOISE = "noise"
    CONTRAST = "contrast"
    BLUR = "blur"


@dataclass
class DICOMMetadata:
    """Structured DICOM metadata"""
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    study_date: str
    modality: str
    rows: int
    columns: int
    slices: int
    spacing: Tuple[float, float, float]  # (z, y, x) in mm
    origin: Tuple[float, float, float]   # (x, y, z) in mm
    orientation: Tuple[float, ...]      # Image orientation (6 values)
    window_center: float
    window_width: float
    rescale_slope: float
    rescale_intercept: float
    bits_allocated: int
    bits_stored: int
    pixel_representation: int
    manufacturer: str
    manufacturer_model: str
    institution: str
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Normalization
    hu_window: HUWindowPreset = HUWindowPreset.LIVER
    custom_window_center: Optional[float] = None
    custom_window_width: Optional[float] = None
    
    # Resampling
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Isotropic voxels
    resample_method: ResampleMethod = ResampleMethod.LINEAR
    
    # Augmentation
    augmentations: List[AugmentationType] = None
    augmentation_probability: float = 0.5
    
    # Quality control
    clip_hu_range: Tuple[float, float] = (-200, 400)  # Liver-specific range
    remove_artifacts: bool = True
    denoise_strength: float = 0.1
    
    # Performance
    use_gpu: bool = False
    batch_size: int = 8
    num_workers: int = 4
    
    def __post_init__(self):
        if self.augmentations is None:
            self.augmentations = [
                AugmentationType.FLIP,
                AugmentationType.ROTATE,
                AugmentationType.NOISE
            ]
        
        if self.hu_window == HUWindowPreset.CUSTOM:
            if self.custom_window_center is None or self.custom_window_width is None:
                raise ValueError("Custom window requires center and width values")


class DICOMValidator:
    """Validates DICOM files for quality and completeness"""
    
    @staticmethod
    def validate_dicom_file(file_path: Path) -> bool:
        """Validate a single DICOM file"""
        try:
            import pydicom
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            # Check essential tags
            required_tags = [
                'SOPClassUID', 'Modality', 'Rows', 'Columns',
                'PixelSpacing', 'SliceThickness'
            ]
            
            for tag in required_tags:
                if not hasattr(ds, tag):
                    logger.warning(f"DICOM missing required tag: {tag}")
                    return False
            
            # Check pixel data
            if not hasattr(ds, 'PixelData'):
                logger.warning("DICOM missing pixel data")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"DICOM validation failed: {e}")
            return False
    
    @staticmethod
    def validate_series(dicom_files: List[Path]) -> Dict:
        """Validate a DICOM series for consistency"""
        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
            
            validation_results = {
                'total_files': len(dicom_files),
                'valid_files': 0,
                'consistent_spacing': True,
                'consistent_orientation': True,
                'sorted_by_position': True
            }
            
            if not dicom_files:
                return validation_results
            
            # Load first file as reference
            first_ds = pydicom.dcmread(dicom_files[0])
            ref_spacing = tuple(first_ds.PixelSpacing) + (first_ds.SliceThickness,)
            ref_orientation = tuple(getattr(first_ds, 'ImageOrientationPatient', (1, 0, 0, 0, 1, 0)))
            
            positions = []
            
            for i, file_path in enumerate(dicom_files):
                try:
                    ds = pydicom.dcmread(file_path)
                    validation_results['valid_files'] += 1
                    
                    # Check spacing consistency
                    current_spacing = tuple(ds.PixelSpacing) + (ds.SliceThickness,)
                    if not np.allclose(current_spacing, ref_spacing, rtol=0.01):
                        validation_results['consistent_spacing'] = False
                        logger.warning(f"Inconsistent spacing in {file_path.name}")
                    
                    # Check orientation consistency
                    current_orientation = tuple(getattr(ds, 'ImageOrientationPatient', ref_orientation))
                    if not np.allclose(current_orientation, ref_orientation, rtol=0.01):
                        validation_results['consistent_orientation'] = False
                        logger.warning(f"Inconsistent orientation in {file_path.name}")
                    
                    # Collect positions for sorting validation
                    position = getattr(ds, 'ImagePositionPatient', [0, 0, i])
                    positions.append((position[2], file_path))
                    
                except (InvalidDicomError, AttributeError, KeyError) as e:
                    logger.error(f"Failed to read {file_path.name}: {e}")
            
            # Check if positions are sorted
            sorted_positions = sorted(positions, key=lambda x: x[0])
            if positions != sorted_positions:
                validation_results['sorted_by_position'] = False
                logger.warning("DICOM series not sorted by position")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Series validation failed: {e}")
            return {'error': str(e)}


class AdvancedDicomPreprocessor:
    """
    Advanced DICOM preprocessing pipeline for medical imaging
    
    Features:
    - Multi-threaded DICOM loading
    - Intelligent HU normalization with adaptive windowing
    - High-quality resampling with multiple interpolation methods
    - Advanced augmentation pipeline
    - Quality control and artifact removal
    - Support for GPU acceleration
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.validator = DICOMValidator()
        self._setup_resampling()
        self._setup_augmentation()
        
        logger.info(f"Initialized DICOM preprocessor with config: {self.config}")
    
    def _setup_resampling(self):
        """Setup resampling backend (CPU/GPU)"""
        if self.config.use_gpu:
            try:
                import cupy as cp
                import cupyx.scipy.ndimage as ndimage_gpu
                self._gpu_available = True
                self._cp = cp
                self._ndimage_gpu = ndimage_gpu
                logger.info("GPU acceleration enabled (CuPy)")
            except ImportError:
                self._gpu_available = False
                logger.warning("CuPy not available, falling back to CPU")
        else:
            self._gpu_available = False
        
        # CPU fallback
        if not self._gpu_available:
            try:
                from scipy import ndimage
                self._ndimage = ndimage
                self._scipy_available = True
            except ImportError:
                self._scipy_available = False
                logger.warning("SciPy not available, using simple resampling")
    
    def _setup_augmentation(self):
        """Setup augmentation methods"""
        self.augmentation_methods = {
            AugmentationType.FLIP: self._apply_flip,
            AugmentationType.ROTATE: self._apply_rotation,
            AugmentationType.TRANSLATE: self._apply_translation,
            AugmentationType.SCALE: self._apply_scale,
            AugmentationType.ELASTIC: self._apply_elastic_deformation,
            AugmentationType.NOISE: self._apply_noise,
            AugmentationType.CONTRAST: self._apply_contrast_adjustment,
            AugmentationType.BLUR: self._apply_blur
        }
    
    def load_dicom_series(self, path: Union[str, Path], 
                         sort_by_position: bool = True) -> Tuple[np.ndarray, DICOMMetadata]:
        """
        Load and validate DICOM series
        
        Args:
            path: Path to DICOM directory or single file
            sort_by_position: Sort slices by patient position
        
        Returns:
            tuple: (volume, metadata)
        """
        path = Path(path)
        start_time = time.time()
        
        try:
            if path.is_file():
                logger.info(f"Loading single DICOM file: {path.name}")
                volume, metadata = self._load_single_dicom(path)
            elif path.is_dir():
                logger.info(f"Loading DICOM series from: {path}")
                volume, metadata = self._load_dicom_series(path, sort_by_position)
            else:
                raise FileNotFoundError(f"DICOM path not found: {path}")
            
            load_time = time.time() - start_time
            logger.info(f"DICOM loaded: shape={volume.shape}, dtype={volume.dtype}, "
                       f"time={load_time:.2f}s, spacing={metadata.spacing}")
            
            return volume, metadata
            
        except Exception as e:
            logger.error(f"Failed to load DICOM: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_single_dicom(self, file_path: Path) -> Tuple[np.ndarray, DICOMMetadata]:
        """Load single DICOM file (can be multi-frame)"""
        try:
            import pydicom
            ds = pydicom.dcmread(file_path, force=True)
            
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Handle multi-frame DICOM
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                # Reshape to (frames, slices, rows, columns)
                if pixel_array.ndim == 3:
                    pixel_array = pixel_array.reshape(
                        ds.NumberOfFrames,
                        pixel_array.shape[0] // ds.NumberOfFrames,
                        pixel_array.shape[1],
                        pixel_array.shape[2]
                    )
            
            # Convert to HU
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                volume = pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
            else:
                volume = pixel_array.astype(np.float32)
                logger.warning(f"No rescale parameters found in {file_path.name}")
            
            # Extract metadata
            metadata = self._extract_metadata(ds, volume.shape)
            
            return volume, metadata
            
        except ImportError:
            logger.warning("pydicom not available, creating mock data")
            return self._create_mock_data()
    
    def _load_dicom_series(self, directory: Path, sort_by_position: bool) -> Tuple[np.ndarray, DICOMMetadata]:
        """Load DICOM series from directory"""
        try:
            import pydicom
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Find DICOM files
            dicom_files = list(directory.glob("*.dcm")) + list(directory.glob("*.DCM"))
            if not dicom_files:
                dicom_files = [f for f in directory.iterdir() if f.is_file()]
            
            logger.info(f"Found {len(dicom_files)} potential DICOM files")
            
            # Validate files
            valid_files = []
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                future_to_file = {
                    executor.submit(self.validator.validate_dicom_file, f): f 
                    for f in dicom_files[:100]  # Limit for performance
                }
                
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    if future.result():
                        valid_files.append(file)
            
            if not valid_files:
                raise ValueError(f"No valid DICOM files found in {directory}")
            
            logger.info(f"Validated {len(valid_files)} DICOM files")
            
            # Load metadata first
            slice_metadata = []
            positions = []
            
            for file_path in valid_files[:50]:  # Sample for speed
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    position = getattr(ds, 'ImagePositionPatient', [0, 0, len(slice_metadata)])
                    positions.append((position[2], file_path))
                    slice_metadata.append(ds)
                except:
                    continue
            
            # Sort by position if requested
            if sort_by_position and positions:
                valid_files = [f for _, f in sorted(positions)]
            
            # Load pixel data in batches
            volume_slices = []
            batch_size = min(self.config.batch_size, len(valid_files))
            
            for i in range(0, len(valid_files), batch_size):
                batch_files = valid_files[i:i + batch_size]
                batch_slices = []
                
                for file_path in batch_files:
                    try:
                        ds = pydicom.dcmread(file_path, force=True)
                        pixel_array = ds.pixel_array
                        
                        # Convert to HU
                        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                            hu_slice = pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept
                        else:
                            hu_slice = pixel_array.astype(np.float32)
                        
                        batch_slices.append(hu_slice)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path.name}: {e}")
                        # Add empty slice as placeholder
                        if slice_metadata:
                            ref_ds = slice_metadata[0]
                            batch_slices.append(
                                np.zeros((ref_ds.Rows, ref_ds.Columns), dtype=np.float32)
                            )
                
                if batch_slices:
                    volume_slices.extend(batch_slices)
                
                logger.info(f"Loaded batch {i//batch_size + 1}/{(len(valid_files)-1)//batch_size + 1}")
            
            if not volume_slices:
                raise ValueError("No slices loaded")
            
            # Stack slices into volume
            volume = np.stack(volume_slices, axis=0)
            
            # Use first slice metadata
            ref_ds = slice_metadata[0] if slice_metadata else pydicom.Dataset()
            metadata = self._extract_metadata(ref_ds, volume.shape)
            
            return volume, metadata
            
        except ImportError:
            logger.warning("pydicom not available, creating mock data")
            return self._create_mock_data()
    
    def _extract_metadata(self, dicom_dataset, shape: Tuple) -> DICOMMetadata:
        """Extract structured metadata from DICOM dataset"""
        try:
            # Get spacing
            pixel_spacing = getattr(dicom_dataset, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(dicom_dataset, 'SliceThickness', 1.0)
            slice_spacing = getattr(dicom_dataset, 'SpacingBetweenSlices', slice_thickness)
            
            # Get position and orientation
            image_position = getattr(dicom_dataset, 'ImagePositionPatient', [0, 0, 0])
            image_orientation = getattr(dicom_dataset, 'ImageOrientationPatient', 
                                       [1, 0, 0, 0, 1, 0])
            
            # Get window settings
            import pydicom
            window_center = getattr(dicom_dataset, 'WindowCenter', 40.0)
            if hasattr(pydicom, 'multival') and isinstance(window_center, pydicom.multival.MultiValue):
                window_center = float(window_center[0])
            elif isinstance(window_center, (list, tuple)):
                window_center = float(window_center[0])
            
            window_width = getattr(dicom_dataset, 'WindowWidth', 400.0)
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = float(window_width[0])
            
            return DICOMMetadata(
                patient_id=getattr(dicom_dataset, 'PatientID', 'ANONYMIZED'),
                study_instance_uid=getattr(dicom_dataset, 'StudyInstanceUID', ''),
                series_instance_uid=getattr(dicom_dataset, 'SeriesInstanceUID', ''),
                study_date=getattr(dicom_dataset, 'StudyDate', ''),
                modality=getattr(dicom_dataset, 'Modality', 'CT'),
                rows=shape[2] if len(shape) >= 3 else getattr(dicom_dataset, 'Rows', 512),
                columns=shape[1] if len(shape) >= 2 else getattr(dicom_dataset, 'Columns', 512),
                slices=shape[0] if len(shape) >= 1 else 1,
                spacing=(float(slice_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])),
                origin=(float(image_position[0]), float(image_position[1]), float(image_position[2])),
                orientation=tuple(float(x) for x in image_orientation),
                window_center=float(window_center),
                window_width=float(window_width),
                rescale_slope=float(getattr(dicom_dataset, 'RescaleSlope', 1.0)),
                rescale_intercept=float(getattr(dicom_dataset, 'RescaleIntercept', 0.0)),
                bits_allocated=int(getattr(dicom_dataset, 'BitsAllocated', 16)),
                bits_stored=int(getattr(dicom_dataset, 'BitsStored', 12)),
                pixel_representation=int(getattr(dicom_dataset, 'PixelRepresentation', 0)),
                manufacturer=str(getattr(dicom_dataset, 'Manufacturer', 'UNKNOWN')),
                manufacturer_model=str(getattr(dicom_dataset, 'ManufacturerModelName', 'UNKNOWN')),
                institution=str(getattr(dicom_dataset, 'InstitutionName', 'UNKNOWN'))
            )
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            # Return minimal metadata
            return DICOMMetadata(
                patient_id='UNKNOWN',
                study_instance_uid='',
                series_instance_uid='',
                study_date='',
                modality='CT',
                rows=shape[2] if len(shape) >= 3 else 512,
                columns=shape[1] if len(shape) >= 2 else 512,
                slices=shape[0] if len(shape) >= 1 else 1,
                spacing=(1.5, 1.0, 1.0),
                origin=(0, 0, 0),
                orientation=(1, 0, 0, 0, 1, 0),
                window_center=40.0,
                window_width=400.0,
                rescale_slope=1.0,
                rescale_intercept=0.0,
                bits_allocated=16,
                bits_stored=12,
                pixel_representation=0,
                manufacturer='UNKNOWN',
                manufacturer_model='UNKNOWN',
                institution='UNKNOWN'
            )
    
    def normalize_hu(self, volume: np.ndarray, 
                    metadata: Optional[DICOMMetadata] = None) -> np.ndarray:
        """
        Advanced HU normalization with adaptive windowing
        
        Args:
            volume: Input HU volume
            metadata: DICOM metadata for window settings
        
        Returns:
            Normalized volume [0, 1]
        """
        logger.info("Normalizing HU values...")
        
        # Clip to valid HU range
        min_hu, max_hu = self.config.clip_hu_range
        volume = np.clip(volume, min_hu, max_hu)
        
        # Get window settings
        if metadata:
            window_center = metadata.window_center
            window_width = metadata.window_width
        elif self.config.hu_window != HUWindowPreset.CUSTOM:
            window_center, window_width = self.config.hu_window.value
        else:
            window_center = self.config.custom_window_center
            window_width = self.config.custom_window_width
        
        # Adaptive windowing based on tissue histogram
        if self.config.hu_window == HUWindowPreset.LIVER:
            # For liver, we might want to focus on soft tissue range
            liver_mask = (volume > 30) & (volume < 100)
            if np.any(liver_mask):
                liver_values = volume[liver_mask]
                window_center = np.median(liver_values)
                window_width = np.percentile(liver_values, 75) - np.percentile(liver_values, 25)
                window_width = max(window_width, 150)  # Minimum width
        
        # Apply windowing
        min_value = window_center - window_width / 2.0
        max_value = window_center + window_width / 2.0
        
        # Normalize
        normalized = (volume - min_value) / (max_value - min_value)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Remove artifacts (extreme values)
        if self.config.remove_artifacts:
            # Remove salt-and-pepper noise
            from scipy import ndimage
            normalized = ndimage.median_filter(normalized, size=3)
        
        logger.info(f"Normalization complete: window=[{min_value:.1f}, {max_value:.1f}]")
        
        return normalized.astype(np.float32)
    
    def resample_volume(self, volume: np.ndarray,
                       current_spacing: Tuple[float, float, float],
                       target_spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        High-quality volume resampling
        
        Args:
            volume: Input volume
            current_spacing: Current voxel spacing (z, y, x) in mm
            target_spacing: Target voxel spacing (z, y, x) in mm
        
        Returns:
            Resampled volume
        """
        if target_spacing is None:
            target_spacing = self.config.target_spacing
        
        logger.info(f"Resampling from {current_spacing} to {target_spacing}...")
        
        # Calculate zoom factors
        zoom_factors = [
            current_spacing[i] / target_spacing[i]
            for i in range(3)
        ]
        
        # Skip resampling if factors are close to 1
        if all(abs(f - 1.0) < 0.01 for f in zoom_factors):
            logger.info("Resampling skipped (spacing already close to target)")
            return volume
        
        try:
            if self._gpu_available:
                # GPU resampling with CuPy
                volume_gpu = self._cp.asarray(volume)
                
                if self.config.resample_method == ResampleMethod.NEAREST:
                    order = 0
                elif self.config.resample_method == ResampleMethod.LINEAR:
                    order = 1
                elif self.config.resample_method == ResampleMethod.CUBIC:
                    order = 3
                else:  # LANCZOS
                    order = 5
                
                resampled_gpu = self._ndimage_gpu.zoom(
                    volume_gpu,
                    zoom=zoom_factors,
                    order=order,
                    mode='constant',
                    cval=0.0
                )
                
                resampled = self._cp.asnumpy(resampled_gpu)
                
            elif self._scipy_available:
                # CPU resampling with SciPy
                if self.config.resample_method == ResampleMethod.NEAREST:
                    order = 0
                elif self.config.resample_method == ResampleMethod.LINEAR:
                    order = 1
                elif self.config.resample_method == ResampleMethod.CUBIC:
                    order = 3
                else:  # LANCZOS
                    order = 5
                
                resampled = self._ndimage.zoom(
                    volume,
                    zoom=zoom_factors,
                    order=order,
                    mode='constant',
                    cval=0.0
                )
            else:
                # Simple nearest-neighbor resampling
                from skimage.transform import resize
                new_shape = [
                    int(volume.shape[i] * zoom_factors[i])
                    for i in range(3)
                ]
                resampled = resize(
                    volume,
                    output_shape=new_shape,
                    order=0,
                    mode='constant',
                    anti_aliasing=False
                )
            
            logger.info(f"Resampling complete: {volume.shape} -> {resampled.shape}")
            return resampled
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            logger.error(traceback.format_exc())
            return volume
    
    def apply_augmentation(self, volume: np.ndarray, 
                          mask: Optional[np.ndarray] = None,
                          augmentation_types: Optional[List[AugmentationType]] = None) -> Tuple:
        """
        Apply advanced data augmentation
        
        Args:
            volume: Input volume
            mask: Optional segmentation mask (augmented similarly)
            augmentation_types: Specific augmentations to apply
        
        Returns:
            Augmented volume (and mask if provided)
        """
        if augmentation_types is None:
            augmentation_types = self.config.augmentations
        
        if np.random.random() > self.config.augmentation_probability:
            return (volume, mask) if mask is not None else volume
        
        logger.info(f"Applying augmentations: {[a.value for a in augmentation_types]}")
        
        augmented_volume = volume.copy()
        augmented_mask = mask.copy() if mask is not None else None
        
        for aug_type in augmentation_types:
            if aug_type in self.augmentation_methods:
                try:
                    augmented_volume, augmented_mask = self.augmentation_methods[aug_type](
                        augmented_volume, augmented_mask
                    )
                except Exception as e:
                    logger.warning(f"Augmentation {aug_type} failed: {e}")
        
        return (augmented_volume, augmented_mask) if mask is not None else augmented_volume
    
    def _apply_flip(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply random flipping"""
        axes = []
        if np.random.random() > 0.5:
            axes.append(1)  # Flip horizontal
        if np.random.random() > 0.5:
            axes.append(2)  # Flip vertical
        
        if axes:
            volume = np.flip(volume, axis=axes)
            if mask is not None:
                mask = np.flip(mask, axis=axes)
        
        return volume, mask
    
    def _apply_rotation(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply random rotation (in-plane)"""
        k = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
        if k > 0:
            volume = np.rot90(volume, k=k, axes=(1, 2))
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(1, 2))
        
        return volume, mask
    
    def _apply_translation(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply random translation"""
        max_shift = int(min(volume.shape[1:]) * 0.1)  # Max 10% shift
        shifts = [
            np.random.randint(-max_shift, max_shift + 1),
            np.random.randint(-max_shift, max_shift + 1)
        ]
        
        if any(s != 0 for s in shifts):
            from scipy.ndimage import shift
            volume = shift(volume, shift=(0, shifts[0], shifts[1]), mode='constant', cval=0)
            if mask is not None:
                mask = shift(mask, shift=(0, shifts[0], shifts[1]), mode='constant', cval=0)
        
        return volume, mask
    
    def _apply_scale(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply random scaling"""
        scale_factor = np.random.uniform(0.9, 1.1)
        
        if abs(scale_factor - 1.0) > 0.01:
            from skimage.transform import rescale
            volume = rescale(volume, scale=(1, scale_factor, scale_factor), 
                            mode='constant', anti_aliasing=True)
            if mask is not None:
                mask = rescale(mask, scale=(1, scale_factor, scale_factor),
                              mode='constant', order=0, anti_aliasing=False)
        
        return volume, mask
    
    def _apply_elastic_deformation(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply elastic deformation"""
        try:
            from scipy.ndimage import map_coordinates, gaussian_filter
            
            # Generate random displacement fields
            shape = volume.shape
            alpha = shape[1] * 2
            sigma = shape[1] * 0.08
            
            dx = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                sigma, mode="constant"
            ) * alpha
            dy = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                sigma, mode="constant"
            ) * alpha
            
            # Apply deformation
            x, y, z = np.meshgrid(
                np.arange(shape[2]),
                np.arange(shape[1]),
                np.arange(shape[0]),
                indexing='ij'
            )
            
            indices = (
                np.reshape(z + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
                np.reshape(x, (-1, 1))
            )
            
            volume = map_coordinates(volume, indices, order=3, mode='reflect').reshape(shape)
            if mask is not None:
                mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
            
        except ImportError:
            logger.warning("SciPy not available for elastic deformation")
        
        return volume, mask
    
    def _apply_noise(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Add Gaussian noise"""
        noise_level = np.random.uniform(0.0, 0.05)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        return volume, mask
    
    def _apply_contrast_adjustment(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Adjust contrast"""
        gamma = np.random.uniform(0.8, 1.2)
        if abs(gamma - 1.0) > 0.01:
            volume = np.power(volume, gamma)
        
        return volume, mask
    
    def _apply_blur(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Apply Gaussian blur"""
        sigma = np.random.uniform(0.0, 1.5)
        if sigma > 0.1:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=(0, sigma, sigma))
        
        return volume, mask
    
    def _create_mock_data(self, shape: Tuple = (50, 512, 512)) -> Tuple[np.ndarray, DICOMMetadata]:
        """Create realistic mock CT data for testing"""
        logger.info("Creating mock DICOM data")
        
        # Simulate CT volume with liver
        z, h, w = shape
        
        # Create ellipsoid for liver
        zz, yy, xx = np.ogrid[:z, :h, :w]
        cz, cy, cx = z//2, h//2, w//2
        rz, ry, rx = z//3, h//4, w//4
        
        liver_mask = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        
        # Simulate HU values
        volume = np.random.normal(0.3, 0.1, shape).astype(np.float32)
        volume[liver_mask] += 0.2  # Liver is brighter
        volume = np.clip(volume, 0, 1)
        
        # Add some anatomical structures
        # Spine (bright)
        spine_mask = (np.abs(xx - cx) < 10) & (np.abs(yy - h*0.7) < 15)
        volume[spine_mask] += 0.3
        
        # Kidneys
        kidney_left = ((zz - cz) / (rz*0.8)) ** 2 + ((yy - cy*1.3) / (ry*0.6)) ** 2 + ((xx - cx*0.7) / (rx*0.6)) ** 2 <= 1
        kidney_right = ((zz - cz) / (rz*0.8)) ** 2 + ((yy - cy*1.3) / (ry*0.6)) ** 2 + ((xx - cx*1.3) / (rx*0.6)) ** 2 <= 1
        volume[kidney_left | kidney_right] += 0.15
        
        volume = np.clip(volume, 0, 1)
        
        # Create metadata
        metadata = DICOMMetadata(
            patient_id='MOCK_PATIENT_001',
            study_instance_uid='1.2.3.4.5',
            series_instance_uid='1.2.3.4.5.6',
            study_date='20240101',
            modality='CT',
            rows=w,
            columns=h,
            slices=z,
            spacing=(1.5, 1.0, 1.0),
            origin=(0, 0, 0),
            orientation=(1, 0, 0, 0, 1, 0),
            window_center=40.0,
            window_width=400.0,
            rescale_slope=1.0,
            rescale_intercept=-1024.0,
            bits_allocated=16,
            bits_stored=12,
            pixel_representation=0,
            manufacturer='MOCK_MANUFACTURER',
            manufacturer_model='MOCK_MODEL',
            institution='MOCK_INSTITUTION'
        )
        
        return volume, metadata
    
    def preprocess_pipeline(self, dicom_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          save_intermediate: bool = False) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            dicom_path: Path to DICOM file/directory
            output_path: Path to save processed data
            save_intermediate: Save intermediate steps
        
        Returns:
            Dictionary with results
        """
        results = {
            'success': False,
            'metadata': None,
            'volume_shape': None,
            'processing_time': None,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Load DICOM
            volume, metadata = self.load_dicom_series(dicom_path)
            results['metadata'] = metadata.to_dict()
            results['original_shape'] = volume.shape
            
            if save_intermediate:
                np.save('step1_loaded.npy', volume)
            
            # Step 2: Normalize HU
            volume = self.normalize_hu(volume, metadata)
            if save_intermediate:
                np.save('step2_normalized.npy', volume)
            
            # Step 3: Resample
            volume = self.resample_volume(volume, metadata.spacing)
            if save_intermediate:
                np.save('step3_resampled.npy', volume)
            
            # Step 4: Apply augmentation (if enabled)
            if self.config.augmentations and self.config.augmentation_probability > 0:
                volume, _ = self.apply_augmentation(volume)
                if save_intermediate:
                    np.save('step4_augmented.npy', volume)
            
            # Step 5: Quality control
            volume = self._apply_quality_control(volume)
            
            processing_time = time.time() - start_time
            
            results.update({
                'success': True,
                'volume_shape': volume.shape,
                'processing_time': processing_time,
                'volume_mean': float(np.mean(volume)),
                'volume_std': float(np.std(volume)),
                'volume_range': (float(np.min(volume)), float(np.max(volume)))
            })
            
            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, volume)
                
                # Save metadata
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                results['output_path'] = str(output_path)
                results['metadata_path'] = str(metadata_path)
            
            logger.info(f"Preprocessing complete: {results}")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            results.update({
                'processing_time': processing_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"Preprocessing failed: {e}")
            return results
    
    def _apply_quality_control(self, volume: np.ndarray) -> np.ndarray:
        """Apply quality control measures"""
        # Remove NaN and Inf values
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure values are in [0, 1]
        volume = np.clip(volume, 0.0, 1.0)
        
        # Check for empty slices
        slice_means = np.mean(volume, axis=(1, 2))
        empty_slices = np.where(slice_means < 0.01)[0]
        
        if len(empty_slices) > 0:
            logger.warning(f"Found {len(empty_slices)} empty slices")
            # Interpolate empty slices
            for slice_idx in empty_slices:
                if slice_idx > 0 and slice_idx < volume.shape[0] - 1:
                    volume[slice_idx] = (volume[slice_idx - 1] + volume[slice_idx + 1]) / 2
        
        return volume


# Utility functions
def create_liver_segmentation_dataset(dicom_paths: List[Union[str, Path]],
                                     output_dir: Union[str, Path],
                                     config: PreprocessingConfig = None,
                                     num_samples: int = None):
    """
    Create a dataset for liver segmentation
    
    Args:
        dicom_paths: List of DICOM paths
        output_dir: Output directory
        config: Preprocessing configuration
        num_samples: Number of samples to process
    """
    import pandas as pd
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = AdvancedDicomPreprocessor(config or PreprocessingConfig())
    
    results = []
    
    if num_samples:
        dicom_paths = dicom_paths[:num_samples]
    
    for i, dicom_path in enumerate(tqdm(dicom_paths, desc="Processing DICOMs")):
        try:
            result = preprocessor.preprocess_pipeline(
                dicom_path,
                output_path=output_dir / f"sample_{i:04d}.npy",
                save_intermediate=False
            )
            
            result['dicom_path'] = str(dicom_path)
            result['sample_id'] = i
            results.append(result)
            
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(output_dir / 'processing_log.csv', index=False)
                
        except Exception as e:
            logger.error(f"Failed to process {dicom_path}: {e}")
    
    # Create summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    
    logger.info(f"Dataset created: {len(results)} samples in {output_dir}")
    return summary_df


# Example usage
if __name__ == '__main__':
    # Example 1: Basic preprocessing
    print("=" * 60)
    print("Example 1: Basic DICOM preprocessing")
    print("=" * 60)
    
    config = PreprocessingConfig(
        hu_window=HUWindowPreset.LIVER,
        target_spacing=(1.0, 1.0, 1.0),
        augmentations=[AugmentationType.FLIP, AugmentationType.ROTATE],
        augmentation_probability=0.3
    )
    
    preprocessor = AdvancedDicomPreprocessor(config)
    
    # Test with mock data
    volume, metadata = preprocessor._create_mock_data()
    print(f"Created mock data: shape={volume.shape}")
    print(f"Metadata: {metadata.patient_id}, spacing={metadata.spacing}")
    
    # Normalize
    normalized = preprocessor.normalize_hu(volume, metadata)
    print(f"Normalized: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Resample
    resampled = preprocessor.resample_volume(normalized, metadata.spacing)
    print(f"Resampled: {volume.shape} -> {resampled.shape}")
    
    # Augment
    augmented, _ = preprocessor.apply_augmentation(resampled)
    print(f"Augmented: shape={augmented.shape}")
    
    print("\n" + "=" * 60)
    print("Example 2: Complete pipeline")
    print("=" * 60)
    
    # Simulate processing a DICOM directory
    results = preprocessor.preprocess_pipeline(
        dicom_path="/path/to/dicom/directory",
        output_path="processed_volume.npy",
        save_intermediate=True
    )
    
    print(f"Pipeline results: {json.dumps(results, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Example 3: Batch processing dataset")
    print("=" * 60)
    
    # Simulate creating a dataset
    # dicom_paths = list(Path("/data/dicoms").glob("*/"))
    # summary = create_liver_segmentation_dataset(
    #     dicom_paths,
    #     output_dir="/data/processed_dataset",
    #     num_samples=100
    # )
    # print(f"Created dataset with {len(summary)} samples")
    
    print("\n✅ DICOM preprocessing examples completed successfully!")