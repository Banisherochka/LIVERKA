# Инструкция по запуску проекта

## Что было сделано

### ✅ Улучшения безопасности
1. Добавлена JWT-аутентификация
2. Валидация файлов (расширение, размер)
3. Защита от path traversal
4. Rate limiting (60 запросов/мин)
5. Логирование безопасности
6. Security headers в ответах

### ✅ Улучшения бекенда
1. Интеграция с реальной нейросетью
2. Улучшенная обработка ошибок
3. Подробное логирование
4. Валидация всех входных данных

### ✅ Улучшения нейросети
1. Обработка ошибок при загрузке DICOM
2. Валидация входных данных
3. Fallback на mock данные при ошибках

## Запуск проекта

### 1. Установка зависимостей бекенда

```bash
cd backend
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

Создайте файл `backend/.env`:

```env
DEBUG=True
SECRET_KEY=your-very-secret-key-here-change-in-production
DATABASE_URL=postgresql://user:password@localhost:5432/liver_segmentation
REDIS_URL=redis://localhost:6379/0
STORAGE_PATH=./storage
MAX_UPLOAD_SIZE=100000000
CORS_ORIGINS=http://localhost:4200
```

**ВАЖНО:** Для production обязательно измените SECRET_KEY!

### 3. Настройка базы данных

Убедитесь, что PostgreSQL запущен, затем:

```bash
cd backend
# Создайте базу данных (если еще не создана)
# psql -U postgres
# CREATE DATABASE liver_segmentation;

# Примените миграции
alembic upgrade head
```

### 4. Запуск бекенда

```bash
cd backend
python run.py
```

Бекенд будет доступен на `http://localhost:8000`

API документация: `http://localhost:8000/docs`

### 5. Установка зависимостей нейросети (опционально)

Если хотите использовать реальную нейросеть:

```bash
cd neural_network/python_services
pip install -r requirements.txt
```

### 6. Запуск фронтенда

```bash
cd frontend
npm install
npm start
```

Фронтенд будет доступен на `http://localhost:4200`

## Проверка работоспособности

### 1. Проверка бекенда

```bash
curl http://localhost:8000/
```

Должен вернуть:
```json
{
  "message": "Liver Segmentation API",
  "version": "1.0.0"
}
```

### 2. Проверка health endpoint

```bash
curl http://localhost:8000/api/v1/health
```

### 3. Вход в систему (создастся дефолтный админ при DEBUG=True)

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"admin\", \"password\": \"admin\"}"
```

Получите токен из ответа и используйте его для дальнейших запросов.

### 4. Загрузка DICOM файла

```bash
curl -X POST http://localhost:8000/api/v1/segmentation/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@path/to/your/dicom/file.dcm"
```

## Структура проекта

```
.
├── backend/              # FastAPI бекенд
│   ├── app/
│   │   ├── core/        # Утилиты безопасности
│   │   ├── middleware/  # Middleware безопасности
│   │   ├── api/         # API endpoints
│   │   ├── services/    # Бизнес-логика
│   │   └── models/      # Модели БД
│   └── run.py           # Точка входа
│
├── frontend/            # Angular фронтенд
│   └── src/
│
└── neural_network/      # Нейросеть для сегментации
    └── python_services/
        └── liver_segmentation/
```

## Важные замечания

1. **Безопасность**: В production обязательно измените SECRET_KEY и настройте CORS правильно
2. **База данных**: Убедитесь, что PostgreSQL запущен перед запуском бекенда
3. **Redis**: Опционально для Celery (фоновые задачи). Без Redis задачи выполнятся синхронно
4. **Файлы**: Загруженные DICOM файлы сохраняются в `backend/storage/`
5. **Логи**: Логи сохраняются в `backend/logs/`

## Возможные проблемы

### Бекенд не запускается
- Проверьте, что PostgreSQL запущен
- Проверьте DATABASE_URL в .env
- Проверьте, что порт 8000 свободен

### Ошибки при загрузке файлов
- Проверьте права на запись в директорию storage
- Убедитесь, что файл имеет расширение .dcm
- Проверьте размер файла (максимум 100MB)

### Нейросеть не работает
- Убедитесь, что установлены зависимости из neural_network/python_services/requirements.txt
- Проверьте, что PyTorch установлен (для GPU или CPU)
- Система автоматически использует mock данные при ошибках

## Дополнительная информация

Подробнее о безопасности см. `SECURITY_IMPROVEMENTS.md`

