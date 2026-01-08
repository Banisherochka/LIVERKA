import { CommonModule } from '@angular/common';
import {
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  Output
} from '@angular/core';
import { ButtonComponent } from '../../shared/ui/button/button.component';
import { UploadFileView } from '../../models/upload-file.model';

@Component({
  selector: 'app-file-dropzone',
  standalone: true,
  imports: [CommonModule, ButtonComponent],
  templateUrl: './file-dropzone.component.html',
  styleUrl: './file-dropzone.component.scss'
})
export class FileDropzoneComponent {
  @Input() title = 'Файлы';
  @Input() files: UploadFileView[] = [];

  @Output() selectFiles = new EventEmitter<FileList>();
  @Output() filesDropped = new EventEmitter<FileList>();
  @Output() launch = new EventEmitter<void>();
  @Output() remove = new EventEmitter<number>();

  isDragging = false;

  constructor(private host: ElementRef<HTMLElement>) {}

  @HostListener('dragover', ['$event'])
  handleDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }

  @HostListener('dragleave', ['$event'])
  handleDragLeave(event: DragEvent) {
    const nextTarget = event.relatedTarget as Node | null;
    if (!nextTarget || !this.host.nativeElement.contains(nextTarget)) {
      this.isDragging = false;
    }
  }

  @HostListener('drop', ['$event'])
  handleDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    const files = event.dataTransfer?.files;
    if (files && files.length) {
      this.filesDropped.emit(files);
    }
  }

  onFileInputChange(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (files && files.length) {
      this.selectFiles.emit(files);
    }
    (event.target as HTMLInputElement).value = '';
  }

  formatSize(size: number) {
    return `${(size / (1024 * 1024)).toFixed(1)} Мб`;
  }
}

