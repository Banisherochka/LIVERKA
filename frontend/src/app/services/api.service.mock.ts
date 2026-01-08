import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { delay } from 'rxjs/operators';

// Mock API сервис для демо без базы данных
@Injectable({
  providedIn: 'root'
})
export class MockApiService {
  private baseUrl = 'http://localhost:8000';

  // Mock login - всегда успешный
  async login(username: string, password: string): Promise<{ access_token: string; token_type: string }> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          access_token: 'mock-jwt-token-for-demo',
          token_type: 'bearer'
        });
      }, 500); // Имитация сетевой задержки
    });
  }

  // Mock upload
  async uploadFile(file: File): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          data: {
            task_id: Math.floor(Math.random() * 1000),
            ct_scan_id: Math.floor(Math.random() * 1000),
            status: 'completed',
            message: 'Mock segmentation completed successfully',
            mock_data: true
          }
        });
      }, 1000);
    });
  }

  // Mock get segmentation result
  async getSegmentationResult(taskId: number): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          id: taskId,
          status: 'completed',
          metrics: {
            dice: 0.934,
            iou: 0.917,
            volume_ml: 1259.5,
            quality: 'Excellent',
            clinical_acceptable: true,
            mock_data: true
          },
          segmentation_file: '/storage/segmentations/seg_mock.nii.gz',
          '3d_model_file': '/storage/3d_models/liver_mock.stl'
        });
      }, 500);
    });
  }

  // Mock health check
  async healthCheck(): Promise<any> {
    return {
      status: 'healthy',
      message: 'Mock API server running',
      version: '1.0.0',
      environment: 'demo_mode',
      mock_data: true
    };
  }

  // Mock 3D model data
  getMock3DModel(): Observable<any> {
    return of({
      geometry: {
        type: 'BoxGeometry',
        parameters: { width: 100, height: 80, depth: 60 }
      },
      material: {
        color: 0x8b4513,
        opacity: 0.7,
        transparent: true
      }
    }).pipe(delay(300));
  }

  // Mock upload progress
  getUploadProgress(): Observable<number> {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
      }
    }, 200);
    
    return of(progress);
  }
}