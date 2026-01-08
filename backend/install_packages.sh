# Offline package installation script
#!/bin/bash

echo "Installing packages with SSL fixes..."

# Update pip first
pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install essential packages individually
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    fastapi uvicorn sqlalchemy alembic psycopg2-binary celery redis \
    passlib[bcrypt] python-jose[cryptography] python-dotenv requests \
    boto3 Pillow pydicom numpy pydantic websockets \
    python-dateutil loguru pytest httpx numpy-stl scikit-image \
    --timeout=300 --retries=10

# If the above fails, try with relaxed version constraints
if [ $? -ne 0 ]; then
    echo "Trying with relaxed version constraints..."
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
        fastapi uvicorn sqlalchemy alembic psycopg2-binary celery redis \
        passlib[bcrypt] python-jose cryptography python-dotenv requests \
        boto3 Pillow pydicom numpy pydantic websockets \
        python-dateutil loguru pytest httpx numpy-stl scikit-image \
        --timeout=300 --retries=10 --no-deps
fi

echo "Installation complete!"