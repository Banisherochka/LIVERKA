# Заметки о переводе с Ruby на Python

## Обзор

Весь код Ruby on Rails приложения был переведен на Python с использованием современных Python библиотек.

## Основные изменения

### Фреймворк
- **Rails** → **FastAPI**
  - Rails контроллеры → FastAPI routes
  - Rails middleware → FastAPI middleware
  - ActionCable (WebSocket) → можно использовать websockets или FastAPI WebSocket

### ORM
- **ActiveRecord** → **SQLAlchemy**
  - Rails модели → SQLAlchemy модели
  - Rails миграции → Alembic миграции
  - Rails associations → SQLAlchemy relationships

### Фоновые задачи
- **GoodJob** → **Celery**
  - Rails jobs → Celery tasks
  - GoodJob cron → Celery beat

### Конфигурация
- **Rails config** → **Pydantic Settings**
  - `config/application.rb` → `app/config.py`
  - Environment variables через `.env` файл

### Сервисы
- **Rails services** → **Python service classes**
  - Паттерн ServiceResult для единообразных результатов
  - OpenStruct → dataclass ServiceResult

## Структура файлов

### Модели (Models)
- `app/models/ct_scan.py` - КТ-сканы
- `app/models/segmentation_task.py` - Задачи сегментации
- `app/models/segmentation_result.py` - Результаты сегментации
- `app/models/administrator.py` - Администраторы
- `app/models/three_d_model.py` - 3D модели
- `app/models/admin_oplog.py` - Логи операций

### Сервисы (Services)
- `app/services/dicom_processing_service.py` - Обработка DICOM файлов
- `app/services/liver_segmentation_service.py` - Сегментация печени
- `app/services/metrics_calculation_service.py` - Расчет метрик
- `app/services/dicom_to_3d_service.py` - Конвертация в 3D

### API (Routes)
- `app/api/v1/health.py` - Health check
- `app/api/v1/segmentations.py` - API сегментации
- `app/api/v1/ct_scans.py` - API КТ-сканов

### Фоновые задачи (Tasks)
- `app/tasks/segmentation_processor.py` - Обработка сегментации

## Основные отличия

### Типы данных
- Ruby symbols (`:pending`) → Python enums (`SegmentationTaskStatus.PENDING`)
- Ruby hashes → Python dictionaries
- Ruby nil → Python None

### Валидация
- Rails validations → Pydantic models (для API) + SQLAlchemy constraints

### Файловое хранилище
- Active Storage → Прямое сохранение файлов в файловую систему
  - Можно добавить S3 через boto3

### Аутентификация
- Rails has_secure_password → passlib с bcrypt

## Запуск

### Разработка
```bash
# API сервер
python run.py

# Celery worker
celery -A app.tasks.celery_app worker --loglevel=info
```

### Production
```bash
# Docker Compose
docker-compose up -d
```

## Миграции

```bash
# Создать миграцию
alembic revision --autogenerate -m "Description"

# Применить миграции
alembic upgrade head
```

## TODO

- [ ] Добавить WebSocket поддержку для обновлений статуса
- [ ] Интегрировать реальный Python сервис нейросети
- [ ] Добавить аутентификацию и авторизацию
- [ ] Добавить тесты (pytest)
- [ ] Настроить логирование
- [ ] Добавить мониторинг и метрики

