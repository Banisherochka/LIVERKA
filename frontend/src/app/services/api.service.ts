import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export const API_BASE_URL = 'http://localhost:8000/api/v1';

export interface SegmentationTask {
  id: number;
  ct_scan_id: number;
  status: string;
  created_at: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = API_BASE_URL;
  private token: string | null = null;

  constructor(private http: HttpClient) {
    // Load token from localStorage
    this.token = localStorage.getItem('auth_token');
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  private getHeaders(): HttpHeaders {
    let headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    if (this.token) {
      headers = headers.set('Authorization', `Bearer ${this.token}`);
    }

    return headers;
  }

  // Authentication
  login(username: string, password: string): Observable<ApiResponse<{ access_token: string; token_type: string }>> {
    return this.http.post<ApiResponse<{ access_token: string; token_type: string }>>(
      `${this.apiUrl}/auth/login`,
      { username, password }
    );
  }

  // Upload DICOM file and start segmentation
  uploadDicom(file: File, patientId?: string): Observable<ApiResponse<{
    task_id: number;
    ct_scan_id: number;
    status: string;
    message: string;
  }>> {
    const formData = new FormData();
    formData.append('file', file);
    if (patientId) {
      formData.append('patient_id', patientId);
    }

    const headers = new HttpHeaders();
    if (this.token) {
      headers.set('Authorization', `Bearer ${this.token}`);
    }

    return this.http.post<ApiResponse<{
      task_id: number;
      ct_scan_id: number;
      status: string;
      message: string;
    }>>(`${this.apiUrl}/segmentation/upload`, formData, { headers });
  }

  // Get segmentation tasks
  getSegmentations(limit: number = 50): Observable<ApiResponse<{ tasks: SegmentationTask[] }>> {
    return this.http.get<ApiResponse<{ tasks: SegmentationTask[] }>>(
      `${this.apiUrl}/segmentations?limit=${limit}`,
      { headers: this.getHeaders() }
    );
  }

  // Get segmentation task details
  getSegmentation(taskId: number): Observable<ApiResponse<any>> {
    return this.http.get<ApiResponse<any>>(
      `${this.apiUrl}/segmentations/${taskId}`,
      { headers: this.getHeaders() }
    );
  }

  // Get segmentation result
  getSegmentationResult(taskId: number): Observable<ApiResponse<any>> {
    return this.http.get<ApiResponse<any>>(
      `${this.apiUrl}/segmentations/${taskId}/result`,
      { headers: this.getHeaders() }
    );
  }

  // Get CT scans
  getCtScans(): Observable<ApiResponse<any[]>> {
    return this.http.get<ApiResponse<any[]>>(
      `${this.apiUrl}/ct_scans`,
      { headers: this.getHeaders() }
    );
  }

  // Get current user info
  getCurrentUser(): Observable<any> {
    return this.http.get(
      `${this.apiUrl}/auth/me`,
      { headers: this.getHeaders() }
    );
  }
}

