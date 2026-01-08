import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { MockApiService } from '../../services/api.service.mock';

@Component({
  selector: 'app-login-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login-page.component.html',
  styleUrls: ['./login-page.component.scss']
})
export class LoginPageComponent {
  username = 'admin';
  password = 'demo123';
  isLoading = false;
  error = '';

  constructor(
    private router: Router,
    private mockApi: MockApiService
  ) {}

  async onSubmit() {
    this.isLoading = true;
    this.error = '';

    try {
      const response = await this.mockApi.login(this.username, this.password);
      
      // Сохраняем mock токен
      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('token_type', response.token_type);
      
      console.log('🎭 Mock login successful');
      this.router.navigate(['/upload']);
      
    } catch (error) {
      console.error('Login error:', error);
      this.error = 'Ошибка входа в систему. Проверьте соединение.';
    } finally {
      this.isLoading = false;
    }
  }

  // Быстрый вход для демо
  quickDemoLogin() {
    localStorage.setItem('access_token', 'demo-token');
    localStorage.setItem('token_type', 'bearer');
    this.router.navigate(['/upload']);
  }
}