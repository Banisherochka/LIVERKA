"""
Security middleware for request validation and rate limiting
"""
import time
from typing import Callable
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.logging_config import app_logger
from app.config import get_settings

settings = get_settings()

# Simple in-memory rate limiter (use Redis in production)
rate_limiter_store = defaultdict(list)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current time
        current_time = time.time()
        
        # Clean old entries
        if client_ip in rate_limiter_store:
            rate_limiter_store[client_ip] = [
                t for t in rate_limiter_store[client_ip]
                if current_time - t < 60
            ]
        
        # Check rate limit
        if len(rate_limiter_store[client_ip]) >= self.requests_per_minute:
            app_logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        rate_limiter_store[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for security monitoring"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = str(request.url.query)
        
        app_logger.info(f"Request: {method} {path} from {client_ip}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log security events
            if response.status_code >= 400:
                app_logger.warning(
                    f"SECURITY: Failed request - {method} {path} "
                    f"Status: {response.status_code} IP: {client_ip} "
                    f"Time: {process_time:.3f}s"
                )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            app_logger.error(
                f"SECURITY: Request error - {method} {path} "
                f"Error: {str(e)} IP: {client_ip} Time: {process_time:.3f}s"
            )
            raise

