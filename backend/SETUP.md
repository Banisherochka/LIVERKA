# Инструкция по настройке проекта

## Шаг 1: Установка зависимостей

Зависимости уже установлены в виртуальное окружение.

## Шаг 2: Настройка базы данных

### Вариант 1: Использование Docker Compose (рекомендуется)

```bash
# Запустить PostgreSQL и Redis
docker-compose up -d db redis
```

### Вариант 2: Локальная установка PostgreSQL

1. Установите PostgreSQL
2. Создайте базу данных:
```sql
CREATE DATABASE liver_segmentation;
```

3. Обновите `.env` файл с правильными учетными данными:
```
DATABASE_URL=postgresql://user:password@localhost:5432/liver_segmentation
```

## Шаг 3: Настройка переменных окружения

Скопируйте `.env.example` в `.env` и настройте:
```bash
cp .env.example .env
# Отредактируйте .env файл
```

## Шаг 4: Создание миграций

```bash
# Активируйте виртуальное окружение
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# или
source venv/bin/activate  # Linux/Mac

# Создайте начальную миграцию
alembic revision --autogenerate -m "Initial migration"

# Примените миграции
alembic upgrade head
```

## Шаг 5: Запуск приложения

### API сервер
```bash
python run.py
# или
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Celery worker (в отдельном терминале)
```bash
celery -A app.tasks.celery_app worker --loglevel=info
```

## Шаг 6: Проверка работы

Откройте в браузере:
- API документация: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

## Полезные команды

### Создание новой миграции
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### Откат миграции
```bash
alembic downgrade -1
```

### Просмотр текущей версии миграции
```bash
alembic current
```

### Просмотр истории миграций
```bash
alembic history
```

