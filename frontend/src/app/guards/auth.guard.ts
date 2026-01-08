import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';
import { Observable, of } from 'rxjs';
import { ApiService } from '../services/api.service';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  canActivate(): Observable<boolean> {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      this.router.navigate(['/login']);
      return of(false);
    }

    // For demo purposes, we'll just check if token exists
    // In a real app, you might want to validate the token with the server
    return of(true);
  }
}
