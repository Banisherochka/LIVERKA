"""
Main FastAPI application
"""
from contextlib import asynccontextmanager
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.router import api_router
from app.database import Base, engine
from app.middleware.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware
)
from app.core.logging_config import app_logger

settings = get_settings()


def custom_json_encoder(obj):
    """Custom JSON encoder for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    app_logger.info("Starting application...")
    
    # Create tables (in production, use Alembic migrations)
    try:
        Base.metadata.create_all(bind=engine)
        app_logger.info("Database tables created/verified")
    except Exception as e:
        app_logger.error(f"Database initialization error: {e}")
    
    # Security check
    if settings.SECRET_KEY == "change-me-in-production" and not settings.DEBUG:
        app_logger.warning("SECURITY: Using default SECRET_KEY in production! Change it immediately!")
    
    app_logger.info(f"CORS origins configured: {settings.CORS_ORIGINS}")
    app_logger.info("Application started successfully")
    yield
    
    # Shutdown
    app_logger.info("Shutting down application...")


# Функция для преобразования строки CORS_ORIGINS в список
def parse_cors_origins(cors_string: str) -> list:
    """Parse CORS origins from string to list"""
    if not cors_string:
        return []
    
    # Если строка начинается с [ и заканчивается ], пробуем парсить как JSON
    cors_string = cors_string.strip()
    if cors_string.startswith('[') and cors_string.endswith(']'):
        try:
            return json.loads(cors_string)
        except:
            pass
    
    # Иначе разбиваем по запятым
    origins = []
    for origin in cors_string.split(','):
        origin = origin.strip()
        if origin:
            origins.append(origin)
    
    return origins if origins else ["http://localhost:4200"]


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    default_response_class=JSONResponse
)

# Security middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware
cors_origins = parse_cors_origins(settings.CORS_ORIGINS)

if "*" in cors_origins and not settings.DEBUG:
    app_logger.warning("SECURITY: CORS allows all origins in production!")
    # In production, restrict to specific origins
    cors_origins = ["http://localhost:4200", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Liver Segmentation API",
        "version": settings.APP_VERSION
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "debug": settings.DEBUG,
        "cors_origins": parse_cors_origins(settings.CORS_ORIGINS)
    }