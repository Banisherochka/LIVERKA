import { Routes } from '@angular/router';
import { UploadPageComponent } from './pages/upload-page/upload-page.component';
import { ResultsPageComponent } from './pages/results-page/results-page.component';
import { LoginPageComponent } from './pages/login-page/login-page.component';
import { AuthGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    component: UploadPageComponent,
    canActivate: [AuthGuard]
  },
  {
    path: 'results/:taskId',
    component: ResultsPageComponent,
    canActivate: [AuthGuard]
  },
  {
    path: 'login',
    component: LoginPageComponent
  },
  {
    path: '**',
    redirectTo: ''
  }
];
