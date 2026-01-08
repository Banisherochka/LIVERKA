import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class MockAuthGuard implements CanActivate {
  constructor(private router: Router) {}

  canActivate(): boolean {
    // Обход через URL параметр demo
    const urlParams = new URLSearchParams(window.location.search);
    const isDemo = urlParams.get('demo') === 'true';
    
    if (isDemo) {
      // Сохраняем mock токен
      localStorage.setItem('access_token', 'demo-token');
      localStorage.setItem('token_type', 'bearer');
      console.log('🎭 Demo mode activated');
      return true;
    }

    // Проверяем наличие токена
    const token = localStorage.getItem('access_token');
    if (token) {
      return true;
    }

    // Если токена нет, показываем логин
    return false;
  }
}