import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-login-page',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './login-page.component.html',
  styleUrl: './login-page.component.scss'
})
export class LoginPageComponent {
  loginForm: FormGroup;
  loading = false;
  error: string | null = null;

  constructor(
    private formBuilder: FormBuilder,
    private apiService: ApiService,
    private router: Router
  ) {
    this.loginForm = this.formBuilder.group({
      username: ['admin', [Validators.required]],
      password: ['admin', [Validators.required]]
    });
  }

  onSubmit() {
    if (this.loginForm.valid) {
      this.loading = true;
      this.error = null;

      const { username, password } = this.loginForm.value;

      this.apiService.login(username, password).subscribe({
        next: (response) => {
          if (response.success && response.data.access_token) {
            // Save token
            this.apiService.setToken(response.data.access_token);
            // Navigate to main page
            this.router.navigate(['/']);
          } else {
            this.error = 'Неверные учетные данные';
            this.loading = false;
          }
        },
        error: (error) => {
          console.error('Login error:', error);
          this.error = 'Ошибка входа в систему. Проверьте соединение.';
          this.loading = false;
        }
      });
    }
  }
}
