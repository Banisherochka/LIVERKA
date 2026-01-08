# Liver Segmentation Backend API (Python)

Python версия backend API для сегментации печени на КТ-сканах.

## Технологический стек

- **FastAPI** - современный веб-фреймворк для Python
- **SQLAlchemy** - ORM для работы с базой данных
- **Alembic** - миграции базы данных
- **Celery** - фоновые задачи
- **PostgreSQL** - база данных
- **Redis** - брокер сообщений для Celery

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл
```

4. Создайте базу данных и выполните миграции:
```bash
# Создайте базу данных PostgreSQL
createdb liver_segmentation

# Выполните миграции
alembic upgrade head
```

## Запуск

### API сервер
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Celery worker (для фоновых задач)
```bash
celery -A app.tasks.celery_app worker --loglevel=info
```

## API Endpoints

### Health Check
- `GET /api/v1/health` - Проверка работоспособности API

### Сегментация
- `POST /api/v1/segmentation/upload` - Загрузка DICOM файла и запуск сегментации
- `POST /api/v1/segmentations` - Создание задачи сегментации для существующего КТ-скана
- `GET /api/v1/segmentations` - Список всех задач сегментации
- `GET /api/v1/segmentations/{id}` - Детали задачи сегментации
- `GET /api/v1/segmentations/{id}/result` - Результаты сегментации с метриками
- `GET /api/v1/segmentations/{id}/download_mask` - Скачать файл маски сегментации

### CT Scans
- `GET /api/v1/ct_scans` - Список всех КТ-сканов
- `GET /api/v1/ct_scans/{id}` - Детали КТ-скана
- `GET /api/v1/ct_scans/{id}/three_d_models` - Список 3D моделей для КТ-скана
- `POST /api/v1/ct_scans/{id}/generate_3d` - Генерация 3D модели из КТ-скана

## Структура проекта

```
backend_python/
├── app/
│   ├── api/              # API routes
│   │   └── v1/          # API v1 endpoints
│   ├── models/          # SQLAlchemy models
│   ├── services/        # Business logic services
│   ├── tasks/           # Celery background tasks
│   ├── config.py        # Configuration
│   ├── database.py      # Database setup
│   └── main.py          # FastAPI application
├── alembic/             # Database migrations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Миграции базы данных

Создание новой миграции:
```bash
alembic revision --autogenerate -m "Description"
```

Применение миграций:
```bash
alembic upgrade head
```

Откат миграции:
```bash
alembic downgrade -1
```

## Разработка

Проект следует структуре Rails приложения, но использует Python экосистему:
- Модели: SQLAlchemy вместо ActiveRecord
- Контроллеры: FastAPI routes вместо Rails controllers
- Сервисы: Python классы с ServiceResult паттерном
- Фоновые задачи: Celery вместо GoodJob

