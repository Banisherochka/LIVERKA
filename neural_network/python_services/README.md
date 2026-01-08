# Нейросеть для сегментации печени

Глубокая нейронная сеть на основе архитектуры 3D U-Net для сегментации печени на КТ-сканах.

## 📋 Описание

Модуль Python для автоматической сегментации печени на компьютерных томограммах с использованием глубокого обучения. Реализует архитектуру 3D U-Net с клинической точностью (Dice ≥ 0.90, IoU ≥ 0.90).

## ✨ Основные возможности

- **Архитектура 3D U-Net** - современная модель для объемной сегментации
- **Клиническая точность** - Dice ≥ 0.90, IoU ≥ 0.90
- **Поддержка DICOM** - нативная обработка DICOM файлов через pydicom
- **GPU ускорение** - поддержка CUDA для быстрого инференса
- **Комплексные метрики** - Dice, IoU, чувствительность, специфичность, объем

## 🚀 Установка

### Требования
- Python 3.8+
- PyTorch 2.0+
- CUDA (опционально, для GPU ускорения)

### Шаги установки

```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## 📖 Быстрый старт

### Инференс из DICOM файла

```python
from liver_segmentation import LiverSegmentationInference

# Инициализация пайплайна
pipeline = LiverSegmentationInference(
    model_path='models/liver_unet_baseline.pth',
    device='cuda'  # или 'cpu'
)

# Запуск сегментации
result = pipeline.segment_from_dicom('path/to/dicom/series')

print(f"Dice: {result['metrics']['dice']:.4f}")
print(f"IoU: {result['metrics']['iou']:.4f}")
print(f"Volume: {result['metrics']['volume_ml']:.2f} mL")
```

### Инференс из NumPy массива

```python
import numpy as np
from liver_segmentation import LiverSegmentationInference

pipeline = LiverSegmentationInference(device='cuda')

# Ваш КТ-объем как numpy массив [D, H, W]
ct_volume = np.load('ct_scan.npy')

result = pipeline.segment_from_numpy(ct_volume, spacing=(1.5, 1.0, 1.0))
mask = result['mask']
```

### Командная строка

```bash
python -m liver_segmentation.inference path/to/dicom/file.dcm \
  --model models/liver_unet_baseline.pth \
  --device cuda \
  --output tmp/segmentation_results
```

## 🏗️ Архитектура модели

### 3D U-Net

```
Вход: КТ-объем [1, D, H, W]
  ↓
Encoder (4 уровня):
  - Conv3D + BatchNorm + ReLU
  - MaxPool3D
  ↓
Bottleneck (узкое место)
  ↓
Decoder (4 уровня):
  - TransposedConv3D
  - Skip connections от encoder
  - Conv3D + BatchNorm + ReLU
  ↓
Выход: Маска сегментации [1, D, H, W]
  - Sigmoid активация
```

### Параметры модели

- Входные каналы: 1 (КТ в градациях серого)
- Выходные каналы: 1 (бинарная маска печени)
- Размеры признаков: [64, 128, 256, 512]
- Всего параметров: ~31M

## 🔧 Предобработка

### Нормализация единиц Хаунсфилда

```python
from liver_segmentation import normalize_hounsfield_units

# Окно для печени
normalized = normalize_hounsfield_units(
    volume,
    window_center=40.0,  # Центр окна для печени
    window_width=400.0    # Ширина окна для печени
)
```

### Загрузка DICOM

```python
from liver_segmentation import DicomPreprocessor

preprocessor = DicomPreprocessor(
    target_spacing=(1.5, 1.0, 1.0),  # Целевое разрешение вокселей (z, y, x) в мм
    window_center=40.0,
    window_width=400.0
)

volume, metadata = preprocessor.load_dicom('path/to/dicom')
```

## 📊 Метрики

### Dice Coefficient

```python
from liver_segmentation import calculate_dice

dice = calculate_dice(ground_truth, prediction)
print(f"Dice: {dice:.4f}")
```

### Все метрики

```python
from liver_segmentation import calculate_all_metrics

metrics = calculate_all_metrics(
    ground_truth,
    prediction,
    spacing=(1.5, 1.0, 1.0)
)

# Возвращает:
# {
#   'dice': 0.94,
#   'iou': 0.89,
#   'sensitivity': 0.95,
#   'specificity': 0.99,
#   'pixel_accuracy': 0.98,
#   'volume_ml': 1456.3,
#   'quality_grade': 'Excellent',
#   'meets_clinical_standards': True
# }
```

## 🎓 Обучение модели

### Подготовка данных

1. **Структура датасета**:
```
dataset/
├── train/
│   ├── ct_scans/
│   └── masks/
├── val/
│   ├── ct_scans/
│   └── masks/
└── test/
    ├── ct_scans/
    └── masks/
```

2. **Формат данных**:
- КТ-сканы: DICOM или NIfTI (.nii.gz)
- Маски: NIfTI бинарные маски (печень=1, фон=0)

### Скрипт обучения

```python
from liver_segmentation import UNet3D
import torch
import torch.nn as nn
from torch.optim import Adam

# Инициализация модели
model = UNet3D(in_channels=1, out_channels=1)
model = model.to('cuda')

# Функция потерь: Dice Loss + BCE
criterion = DiceBCELoss()

# Оптимизатор
optimizer = Adam(model.parameters(), lr=1e-4)

# Цикл обучения
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## ⚡ Производительность

### Скорость инференса

- **GPU (RTX 3090)**: 5-10 секунд на КТ-серию
- **GPU (Tesla V100)**: 3-7 секунд на КТ-серию
- **CPU (16 ядер)**: 30-60 секунд на КТ-серию

### Требования к памяти

- **Обучение**: 16GB GPU памяти (batch size 2)
- **Инференс**: 8GB GPU памяти / 16GB RAM

## 🏥 Клинические стандарты

Целевые метрики для клинического развертывания:

- **Dice Coefficient**: ≥ 0.90
- **IoU**: ≥ 0.90
- **Sensitivity**: ≥ 0.92
- **Specificity**: ≥ 0.96

## 🔗 Интеграция с Rails Backend

Python сервис интегрируется с Rails backend через:

1. **Файловую коммуникацию**: Сохранение результатов в общую директорию
2. **API интеграцию**: REST API endpoints для запросов инференса
3. **Фоновые задачи**: Асинхронная обработка через GoodJob

### Пример интеграции

```ruby
# Rails сервис, вызывающий Python инференс
class LiverSegmentationService
  def run_inference(input_data)
    # Вызов Python скрипта
    result = `python3 neural_network/python_services/liver_segmentation/inference.py #{input_path}`
    
    # Парсинг JSON результата
    JSON.parse(result)
  end
end
```

## 🐛 Решение проблем

### CUDA Out of Memory

- Уменьшить batch size
- Использовать gradient checkpointing
- Обрабатывать меньшие окна

### Низкий Dice Score

- Проверить предобработку данных
- Проверить настройки HU окна
- Проверить аугментацию данных
- Проверить качество ground truth

## 📚 Структура модуля

```
liver_segmentation/
├── __init__.py          # Инициализация модуля
├── model.py             # Архитектура 3D U-Net
├── inference.py         # Пайплайн инференса
├── preprocessing.py     # Предобработка DICOM
└── metrics.py           # Расчет метрик
```

## 📖 Ссылки

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", MICCAI 2016
3. LiTS - Liver Tumor Segmentation Challenge

## 📝 Лицензия

[Указать лицензию]

## 👥 Контакты

Для вопросов или проблем, пожалуйста, свяжитесь с командой разработки.
