"""
Comprehensive test script for liver segmentation pipeline
Tests: DICOM upload, segmentation, metrics, 3D model generation
"""
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DICOM_TEST_FILE = "/home/runner/app/Anon_Liver/Abdomen_(+C) - 19053/VENOUS_125mm_7/IM-0001-0050.dcm"
DICOM_FOLDER = "/home/runner/app/Anon_Liver/Abdomen_(+C) - 19053/VENOUS_125mm_7"


class LiverSegmentationTester:
    """Test class for liver segmentation pipeline"""
    
    def __init__(self):
        self.api_url = API_BASE_URL
        self.test_results = {
            "api_health": False,
            "dicom_upload": False,
            "segmentation_task": False,
            "segmentation_result": False,
            "metrics_calculation": False,
            "3d_model_generation": False
        }
        self.ct_scan_id = None
        self.task_id = None
    
    def print_header(self, text: str):
        """Print section header"""
        print("\n" + "=" * 80)
        print(text.center(80))
        print("=" * 80 + "\n")
    
    def print_step(self, text: str):
        """Print step description"""
        print(f"\n{'─' * 80}")
        print(f"🔹 {text}")
        print(f"{'─' * 80}\n")
    
    def test_api_health(self) -> bool:
        """Test 1: Check API health"""
        self.print_step("TEST 1: API Health Check")
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ API is healthy")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                print(f"   Version: {data.get('version')}")
                self.test_results["api_health"] = True
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ API health check error: {e}")
            return False
    
    def test_dicom_upload(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Test 2: Upload DICOM file"""
        self.print_step("TEST 2: DICOM File Upload")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return None
            
            print(f"📁 Uploading file: {file_path.name}")
            print(f"   Size: {file_path.stat().st_size / 1024:.2f} KB")
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/dicom')}
                data = {'patient_id': '19053'}
                
                response = requests.post(
                    f"{self.api_url}/segmentation/upload",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ DICOM upload successful")
                print(f"   CT Scan ID: {result['data']['ct_scan_id']}")
                print(f"   Task ID: {result['data']['task_id']}")
                print(f"   Status: {result['data']['status']}")
                
                self.ct_scan_id = result['data']['ct_scan_id']
                self.task_id = result['data']['task_id']
                self.test_results["dicom_upload"] = True
                return result['data']
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
    
    def test_segmentation_status(self, task_id: int, wait_for_completion: bool = True) -> Optional[Dict[str, Any]]:
        """Test 3: Check segmentation task status"""
        self.print_step("TEST 3: Segmentation Task Status")
        
        try:
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts:
                response = requests.get(f"{self.api_url}/segmentations/{task_id}", timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    task_data = result['data']
                    
                    status = task_data['status']
                    print(f"📊 Task {task_id} status: {status}")
                    
                    if status == 'completed':
                        print("✅ Segmentation completed successfully")
                        print(f"   Inference time: {task_data.get('inference_time_ms', 0)} ms")
                        self.test_results["segmentation_task"] = True
                        return task_data
                    elif status == 'failed':
                        print(f"❌ Segmentation failed: {task_data.get('error_message', 'Unknown error')}")
                        return None
                    elif status in ['pending', 'processing']:
                        if wait_for_completion:
                            print(f"⏳ Waiting for completion... (attempt {attempt + 1}/{max_attempts})")
                            time.sleep(2)
                            attempt += 1
                        else:
                            return task_data
                    else:
                        print(f"❓ Unknown status: {status}")
                        return None
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return None
            
            print(f"⏱️ Timeout waiting for segmentation completion")
            return None
            
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return None
    
    def test_segmentation_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Test 4: Get segmentation result with metrics"""
        self.print_step("TEST 4: Segmentation Result & Metrics")
        
        try:
            response = requests.get(f"{self.api_url}/segmentations/{task_id}/result", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print("✅ Segmentation result retrieved")
                print(f"\n📈 Metrics:")
                metrics = data.get('metrics', {})
                
                if metrics.get('dice') is not None:
                    print(f"   Dice Coefficient: {metrics['dice']:.4f}")
                if metrics.get('iou') is not None:
                    print(f"   IoU Score: {metrics['iou']:.4f}")
                if metrics.get('volume_ml') is not None:
                    print(f"   Liver Volume: {metrics['volume_ml']:.2f} ml")
                if metrics.get('quality_grade'):
                    print(f"   Quality Grade: {metrics['quality_grade']}")
                if metrics.get('meets_clinical_standards') is not None:
                    print(f"   Meets Clinical Standards: {metrics['meets_clinical_standards']}")
                
                self.test_results["segmentation_result"] = True
                self.test_results["metrics_calculation"] = True
                return data
            else:
                print(f"❌ Failed to get result: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Result retrieval error: {e}")
            return None
    
    def test_3d_model_generation(self, ct_scan_id: int) -> Optional[Dict[str, Any]]:
        """Test 5: Generate 3D model"""
        self.print_step("TEST 5: 3D Model Generation")
        
        try:
            print(f"🎨 Generating 3D model for CT scan {ct_scan_id}...")
            
            response = requests.post(
                f"{self.api_url}/ct_scans/{ct_scan_id}/generate_3d",
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print("✅ 3D model generated successfully")
                print(f"   Model ID: {data.get('id')}")
                print(f"   Model Name: {data.get('name')}")
                print(f"   Status: {data.get('status')}")
                
                self.test_results["3d_model_generation"] = True
                return data
            else:
                print(f"❌ 3D generation failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 3D generation error: {e}")
            return None
    
    def test_list_segmentations(self):
        """Test 6: List all segmentations"""
        self.print_step("TEST 6: List All Segmentations")
        
        try:
            response = requests.get(f"{self.api_url}/segmentations", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                tasks = result['data']['tasks']
                
                print(f"✅ Retrieved {len(tasks)} segmentation tasks")
                
                for i, task in enumerate(tasks[:5], 1):
                    print(f"\n   Task {i}:")
                    print(f"      ID: {task['id']}")
                    print(f"      Status: {task['status']}")
                    print(f"      CT Scan ID: {task.get('ct_scan_id')}")
                    print(f"      Created: {task.get('created_at', 'N/A')[:19]}")
                
                return True
            else:
                print(f"❌ Failed to list segmentations: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ List segmentations error: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%\n")
        
        print("Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}  {test_name.replace('_', ' ').title()}")
        
        print("\n" + "=" * 80)
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        self.print_header("LIVER SEGMENTATION PIPELINE - FULL TEST SUITE")
        
        print("🚀 Starting comprehensive test suite...")
        print(f"📍 API URL: {self.api_url}")
        print(f"📁 Test File: {Path(DICOM_TEST_FILE).name}")
        
        # Test 1: Health check
        if not self.test_api_health():
            print("\n⚠️ API is not available. Please start the server first.")
            return False
        
        # Test 2: Upload DICOM
        upload_result = self.test_dicom_upload(DICOM_TEST_FILE)
        if not upload_result:
            print("\n⚠️ DICOM upload failed. Cannot continue tests.")
            self.print_summary()
            return False
        
        # Test 3: Check segmentation status
        status_result = self.test_segmentation_status(self.task_id, wait_for_completion=True)
        if not status_result:
            print("\n⚠️ Segmentation task failed or timed out.")
            self.print_summary()
            return False
        
        # Test 4: Get segmentation result
        result_data = self.test_segmentation_result(self.task_id)
        if not result_data:
            print("\n⚠️ Could not retrieve segmentation results.")
        
        # Test 5: Generate 3D model
        if self.ct_scan_id:
            self.test_3d_model_generation(self.ct_scan_id)
        
        # Test 6: List all segmentations
        self.test_list_segmentations()
        
        # Print summary
        self.print_summary()
        
        return all(self.test_results.values())


def main():
    """Main test function"""
    tester = LiverSegmentationTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
