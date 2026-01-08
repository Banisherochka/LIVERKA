"""
Test script to analyze DICOM files in Anon_Liver folder
"""
import sys
from pathlib import Path
import pydicom
import numpy as np

def analyze_dicom_structure():
    """Analyze DICOM structure in Anon_Liver folder"""
    
    anon_liver_path = Path("/home/runner/app/Anon_Liver")
    
    print("=" * 80)
    print("DICOM DATA ANALYSIS - Anon_Liver Folder")
    print("=" * 80)
    print()
    
    # Find all DICOM directories
    study_dirs = [d for d in anon_liver_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(study_dirs)} study directories:")
    for study_dir in study_dirs:
        print(f"  - {study_dir.name}")
    print()
    
    # Analyze each study
    for study_dir in study_dirs:
        print("-" * 80)
        print(f"Study: {study_dir.name}")
        print("-" * 80)
        
        # Find series directories
        series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
        print(f"Series count: {len(series_dirs)}")
        print()
        
        for series_dir in series_dirs:
            print(f"  Series: {series_dir.name}")
            
            # Count DICOM files
            dicom_files = sorted(list(series_dir.glob("*.dcm")))
            print(f"    DICOM files: {len(dicom_files)}")
            
            if dicom_files:
                # Read first and last file
                first_file = dicom_files[0]
                last_file = dicom_files[-1]
                
                try:
                    # Read first file metadata
                    ds = pydicom.dcmread(first_file)
                    
                    print(f"    Patient ID: {ds.PatientID if hasattr(ds, 'PatientID') else 'N/A'}")
                    print(f"    Study Date: {ds.StudyDate if hasattr(ds, 'StudyDate') else 'N/A'}")
                    print(f"    Modality: {ds.Modality if hasattr(ds, 'Modality') else 'N/A'}")
                    print(f"    Series Description: {ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'}")
                    
                    # Get image dimensions
                    pixel_array = ds.pixel_array
                    print(f"    Image Size: {pixel_array.shape}")
                    
                    # Get spacing information
                    if hasattr(ds, 'PixelSpacing'):
                        pixel_spacing = ds.PixelSpacing
                        print(f"    Pixel Spacing: {pixel_spacing[0]:.4f} x {pixel_spacing[1]:.4f} mm")
                    
                    if hasattr(ds, 'SliceThickness'):
                        print(f"    Slice Thickness: {ds.SliceThickness} mm")
                    elif hasattr(ds, 'SpacingBetweenSlices'):
                        print(f"    Spacing Between Slices: {ds.SpacingBetweenSlices} mm")
                    
                    # Get HU range
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope)
                        intercept = float(ds.RescaleIntercept)
                        hu_array = pixel_array * slope + intercept
                        print(f"    Rescale Slope: {slope}")
                        print(f"    Rescale Intercept: {intercept}")
                        print(f"    HU Range: [{hu_array.min():.1f}, {hu_array.max():.1f}]")
                    
                    # Calculate volume information
                    total_slices = len(dicom_files)
                    if hasattr(ds, 'PixelSpacing') and hasattr(ds, 'SliceThickness'):
                        voxel_volume = float(ds.PixelSpacing[0]) * float(ds.PixelSpacing[1]) * float(ds.SliceThickness)
                        total_volume_cm3 = (pixel_array.shape[0] * pixel_array.shape[1] * total_slices * voxel_volume) / 1000
                        print(f"    Total Volume: {total_volume_cm3:.2f} cm³")
                    
                    print(f"    Total Slices: {total_slices}")
                    
                except Exception as e:
                    print(f"    Error reading DICOM: {e}")
            
            print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_dicom_structure()
