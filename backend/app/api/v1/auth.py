"""
Authentication endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import timedelta

from app.api.dependencies import get_database
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token
)
from app.models.administrator import Administrator
from app.config import get_settings

settings = get_settings()
router = APIRouter()
security_scheme = HTTPBearer()


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_database)
):
    """Login endpoint"""
    # Find administrator
    admin = db.query(Administrator).filter(
        Administrator.username == login_data.username
    ).first()
    
    if not admin:
        # Create default admin if none exists (only for development)
        if settings.DEBUG and login_data.username == "admin":
            admin = Administrator(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin")
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
    
    # Verify password
    if not verify_password(login_data.password, admin.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data={"sub": admin.username, "admin_id": admin.id},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/auth/me")
async def get_current_user(
    payload: dict = Depends(verify_token),
    db: Session = Depends(get_database)
):
    """Get current user info"""
    username = payload.get("sub")
    admin = db.query(Administrator).filter(
        Administrator.username == username
    ).first()
    
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": admin.id,
        "username": admin.username,
        "email": admin.email,
        "created_at": admin.created_at.isoformat() if admin.created_at else None
    }

