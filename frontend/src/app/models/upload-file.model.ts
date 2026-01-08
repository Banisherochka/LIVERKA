export type UploadStatus = 'pending' | 'uploaded' | 'processing' | 'completed' | 'error' | 'cancelled';

export interface UploadFileView {
  id: number;
  order: number;
  name: string;
  size: number;
  status: UploadStatus;
  file?: File;
  taskId?: number;
}

