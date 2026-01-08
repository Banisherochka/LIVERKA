import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatTabsModule } from '@angular/material/tabs';
import { ApiService } from '../../services/api.service';
import { ThreeDViewerComponent } from '../../components/3d-viewer/3d-viewer.component';

@Component({
  selector: 'app-results-page',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    MatCardModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatChipsModule,
    MatTabsModule,
    ThreeDViewerComponent
  ],
  templateUrl: './results-page.component.html',
  styleUrl: './results-page.component.scss'
})
export class ResultsPageComponent implements OnInit {
  taskId: number | null = null;
  loading = true;
  error: string | null = null;
  result: any = null;
  selectedTabIndex = 0;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.taskId = +params['taskId'];
      if (this.taskId) {
        this.loadResults();
      }
    });
  }

  loadResults() {
    if (!this.taskId) return;

    this.apiService.getSegmentationResult(this.taskId).subscribe({
      next: (response) => {
        if (response.success) {
          this.result = response.data;
          this.loading = false;
        } else {
          this.error = 'Ошибка загрузки результатов';
          this.loading = false;
        }
      },
      error: (error) => {
        console.error('Error loading results:', error);
        this.error = 'Ошибка загрузки результатов';
        this.loading = false;
      }
    });
  }

  getQualityColor(grade: string): string {
    switch (grade?.toLowerCase()) {
      case 'excellent': return 'primary';
      case 'good': return 'accent';
      case 'fair': return 'warn';
      default: return 'default';
    }
  }

  downloadMask() {
    if (this.taskId) {
      const url = `${this.apiService['apiUrl']}/segmentations/${this.taskId}/download_mask`;
      window.open(url, '_blank');
    }
  }

  switch3DView() {
    this.selectedTabIndex = 1; // Switch to 3D viewer tab
  }

  goBack() {
    this.router.navigate(['/']);
  }
}
