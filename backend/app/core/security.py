"""
Security utilities for authentication and authorization
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
            
        return payload
    except JWTError:
        raise credentials_exception


def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token"""
    return secrets.token_urlsafe(length)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    import os
    from pathlib import Path
    
    # Remove directory components
    filename = os.path.basename(filename)
    
    # Remove any path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    return filename


def validate_file_extension(filename: str, allowed_extensions: list = None) -> bool:
    """Validate file extension"""
    from pathlib import Path
    
    if allowed_extensions is None:
        allowed_extensions = ['.dcm', '.dicom', '.DCM', '.DICOM']
    
    ext = Path(filename).suffix.lower()
    return ext in [e.lower() for e in allowed_extensions]


def validate_file_size(file_size: int, max_size: int = None) -> bool:
    """Validate file size"""
    if max_size is None:
        max_size = settings.MAX_UPLOAD_SIZE
    
    return file_size <= max_size

