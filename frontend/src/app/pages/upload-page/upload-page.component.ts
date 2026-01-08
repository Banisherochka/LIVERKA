import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FileDropzoneComponent } from '../../components/file-dropzone/file-dropzone.component';
import { ButtonComponent } from '../../shared/ui/button/button.component';
import { UploadFileView, UploadStatus } from '../../models/upload-file.model';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-upload-page',
  standalone: true,
  imports: [CommonModule, ButtonComponent, FileDropzoneComponent],
  templateUrl: './upload-page.component.html',
  styleUrl: './upload-page.component.scss'
})
export class UploadPageComponent {
  files: UploadFileView[] = [];
  private counter = 1;
  uploading = false;
  uploadProgress: { [key: number]: number } = {};

  constructor(private apiService: ApiService) {}

  handleFilesSelected(fileList: FileList) {
    const items = Array.from(fileList).map((file) => ({
      id: this.counter++,
      order: this.files.length + 1,
      name: file.name,
      size: file.size,
      status: 'pending' as UploadStatus,
      file: file
    }));
    this.files = [...this.files, ...items];
  }

  handleLaunch() {
    // Upload all pending files
    const pendingFiles = this.files.filter(f => f.status === 'pending' || f.status === 'uploaded');
    
    pendingFiles.forEach(fileView => {
      if (fileView.file) {
        this.uploadFile(fileView);
      }
    });
  }

  private uploadFile(fileView: UploadFileView) {
    if (!fileView.file) return;

    this.uploading = true;
    fileView.status = 'processing';

    this.apiService.uploadDicom(fileView.file).subscribe({
      next: (response) => {
        if (response.success) {
          fileView.status = 'uploaded';
          fileView.taskId = response.data.task_id;
          // Optionally poll for segmentation results
          this.pollSegmentationStatus(response.data.task_id, fileView.id);
        } else {
          fileView.status = 'error';
        }
        this.uploading = false;
      },
      error: (error) => {
        console.error('Upload error:', error);
        fileView.status = 'error';
        this.uploading = false;
      }
    });
  }

  private pollSegmentationStatus(taskId: number, fileId: number) {
    const interval = setInterval(() => {
      this.apiService.getSegmentation(taskId).subscribe({
        next: (response) => {
          if (response.success) {
            const task = response.data;
            const fileView = this.files.find(f => f.id === fileId);
            
            if (fileView) {
              if (task.status === 'completed') {
                fileView.status = 'completed';
                clearInterval(interval);
              } else if (task.status === 'failed') {
                fileView.status = 'error';
                clearInterval(interval);
              }
            }
          }
        },
        error: () => clearInterval(interval)
      });
    }, 2000); // Poll every 2 seconds

    // Stop polling after 5 minutes
    setTimeout(() => clearInterval(interval), 300000);
  }

  handleRemove(id: number) {
    this.files = this.files.filter((file) => file.id !== id).map((file, index) => ({
      ...file,
      order: index + 1
    }));
  }
}

