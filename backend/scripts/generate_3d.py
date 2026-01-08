#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import pydicom
from stl import mesh
from skimage import measure

def load_dicom_series(dicom_dir):
    """Загружает серию DICOM файлов"""
    dicom_files = []
    
    for filename in sorted(os.listdir(dicom_dir)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(dicom_dir, filename)
            dicom = pydicom.dcmread(filepath)
            dicom_files.append(dicom)
    
    return dicom_files

def create_volume_from_dicom(dicom_files):
    """Создает 3D volume из DICOM срезов"""
    # Сортируем по позиции среза
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Получаем размеры
    rows = dicom_files[0].Rows
    cols = dicom_files[0].Columns
    slices = len(dicom_files)
    
    # Создаем 3D массив
    volume = np.zeros((slices, rows, cols), dtype=np.int16)
    
    for i, dicom in enumerate(dicom_files):
        volume[i, :, :] = dicom.pixel_array
    
    return volume

def generate_stl(volume, threshold=200):
    """Генерирует STL модель из volume"""
    # Применяем порог
    binary_volume = volume > threshold
    
    # Используем marching cubes для генерации поверхности
    verts, faces, normals, values = measure.marching_cubes(
        binary_volume, 
        level=0.5,
        spacing=(1.0, 1.0, 1.0)
    )
    
    # Создаем STL mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]
    
    return stl_mesh

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python generate_3d.py <dicom_dir>"}))
        sys.exit(1)
    
    dicom_dir = sys.argv[1]
    
    try:
        # 1. Загружаем DICOM файлы
        dicom_files = load_dicom_series(dicom_dir)
        
        if not dicom_files:
            raise ValueError("No DICOM files found")
        
        # 2. Создаем volume
        volume = create_volume_from_dicom(dicom_files)
        
        # 3. Генерируем STL
        stl_mesh = generate_stl(volume)
        
        # 4. Сохраняем STL файл
        output_dir = os.path.join(os.path.dirname(dicom_dir), '..', 'storage', '3d_models')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'generated_model.stl')
        stl_mesh.save(output_path)
        
        # 5. Возвращаем результат
        result = {
            "success": True,
            "stl_path": output_path,
            "dimensions": volume.shape,
            "message": "3D model generated successfully"
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()

