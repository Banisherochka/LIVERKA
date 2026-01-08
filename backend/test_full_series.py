"""
Advanced test script for full DICOM series processing
Tests complete workflow with 393 DICOM slices
"""
import sys
import time
import requests
from pathlib import Path

API_BASE_URL = "http://localhost:8000/api/v1"
DICOM_SERIES_DIR = "/home/runner/app/Anon_Liver/Abdomen_(+C) - 19053/VENOUS_125mm_7"


def print_header(text: str):
    """Print section header"""
    print("\n" + "=" * 100)
    print(text.center(100))
    print("=" * 100 + "\n")


def test_full_series_upload():
    """Test uploading a full DICOM series"""
    print_header("FULL DICOM SERIES PROCESSING TEST")
    
    series_dir = Path(DICOM_SERIES_DIR)
    dicom_files = sorted(list(series_dir.glob("*.dcm")))
    
    print(f"📂 DICOM Series Directory: {series_dir.name}")
    print(f"📊 Total DICOM files found: {len(dicom_files)}")
    print(f"📁 First file: {dicom_files[0].name}")
    print(f"📁 Last file: {dicom_files[-1].name}")
    print(f"💾 Total size: {sum(f.stat().st_size for f in dicom_files) / 1024 / 1024:.2f} MB\n")
    
    # Test with middle slice (most likely to contain liver)
    test_file = dicom_files[len(dicom_files) // 2]
    
    print(f"🎯 Selected test file: {test_file.name} (middle of series)")
    print(f"📏 File size: {test_file.stat().st_size / 1024:.2f} KB\n")
    
    print("─" * 100)
    print("STEP 1: Uploading DICOM file...")
    print("─" * 100)
    
    try:
        start_time = time.time()
        
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/dicom')}
            data = {'patient_id': '19053'}
            
            response = requests.post(
                f"{API_BASE_URL}/segmentation/upload",
                files=files,
                data=data,
                timeout=120
            )
        
        upload_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful ({upload_time:.2f}s)")
            print(f"   CT Scan ID: {result['data']['ct_scan_id']}")
            print(f"   Task ID: {result['data']['task_id']}")
            print(f"   Initial Status: {result['data']['status']}\n")
            
            ct_scan_id = result['data']['ct_scan_id']
            task_id = result['data']['task_id']
            
            # Step 2: Wait for segmentation completion
            print("─" * 100)
            print("STEP 2: Waiting for segmentation to complete...")
            print("─" * 100)
            
            completed = False
            attempt = 0
            max_attempts = 60
            
            while not completed and attempt < max_attempts:
                response = requests.get(f"{API_BASE_URL}/segmentations/{task_id}")
                
                if response.status_code == 200:
                    task_data = response.json()['data']
                    status = task_data['status']
                    
                    if status == 'completed':
                        print(f"✅ Segmentation completed!")
                        print(f"   Inference time: {task_data.get('inference_time_ms', 0)} ms\n")
                        completed = True
                    elif status == 'failed':
                        print(f"❌ Segmentation failed: {task_data.get('error_message')}")
                        return False
                    else:
                        print(f"⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})", end='\r')
                        time.sleep(2)
                        attempt += 1
                else:
                    print(f"❌ Error checking status: {response.status_code}")
                    return False
            
            if not completed:
                print(f"\n⏱️ Timeout after {max_attempts * 2}s")
                return False
            
            # Step 3: Get segmentation results
            print("─" * 100)
            print("STEP 3: Retrieving segmentation results...")
            print("─" * 100)
            
            response = requests.get(f"{API_BASE_URL}/segmentations/{task_id}/result")
            
            if response.status_code == 200:
                result_data = response.json()['data']
                metrics = result_data.get('metrics', {})
                
                print("✅ Results retrieved successfully\n")
                print("📊 SEGMENTATION METRICS:")
                print(f"   • Dice Coefficient: {metrics.get('dice', 0):.4f}")
                print(f"   • IoU Score: {metrics.get('iou', 0):.4f}")
                print(f"   • Liver Volume: {metrics.get('volume_ml', 0):.2f} ml ({metrics.get('volume_ml', 0) / 1000:.3f} L)")
                print(f"   • Quality Grade: {metrics.get('quality_grade', 'N/A')}")
                print(f"   • Clinical Standards: {'✅ Met' if metrics.get('meets_clinical_standards') else '❌ Not Met'}\n")
            else:
                print(f"❌ Failed to retrieve results: {response.status_code}\n")
            
            # Step 4: Generate 3D model
            print("─" * 100)
            print("STEP 4: Generating 3D model...")
            print("─" * 100)
            
            start_3d = time.time()
            response = requests.post(
                f"{API_BASE_URL}/ct_scans/{ct_scan_id}/generate_3d",
                timeout=180
            )
            
            time_3d = time.time() - start_3d
            
            if response.status_code == 200:
                model_data = response.json()['data']
                print(f"✅ 3D model generated successfully ({time_3d:.2f}s)")
                print(f"   Model ID: {model_data.get('id')}")
                print(f"   Model Name: {model_data.get('name')}")
                print(f"   Model File: {model_data.get('model_file')}\n")
            else:
                print(f"❌ 3D generation failed: {response.status_code}")
                print(f"   Error: {response.text}\n")
            
            # Step 5: Summary
            print_header("TEST SUMMARY")
            total_time = time.time() - start_time
            print(f"⏱️  Total Processing Time: {total_time:.2f}s")
            print(f"📤 Upload Time: {upload_time:.2f}s")
            print(f"🧠 Segmentation Time: {task_data.get('inference_time_ms', 0) / 1000:.2f}s")
            print(f"🎨 3D Generation Time: {time_3d:.2f}s")
            print(f"📊 Throughput: {test_file.stat().st_size / 1024 / upload_time:.2f} KB/s\n")
            
            print("✅ ALL TESTS PASSED!")
            print("=" * 100)
            
            return True
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_series_upload()
    sys.exit(0 if success else 1)
