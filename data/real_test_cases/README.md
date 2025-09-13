# Real Test Cases - Реальные Тестовые Случаи

Эта папка содержит реальные изображения автомобилей для тестирования обученной модели без аннотаций.

## Структура

```
real_test_cases/
├── production/          # Реальные кейсы из продакшена
│   ├── case_001.jpg
│   ├── case_002.jpg
│   └── ...
├── validation/          # Валидационные кейсы (можно добавить Ground Truth позже)
│   ├── val_001.jpg
│   ├── val_002.jpg
│   └── ...
├── results/            # Результаты инференса
│   ├── production_results/
│   └── validation_results/
└── metadata.json       # Метаданные о тестовых кейсах
```

## Использование

### 1. Добавление реальных изображений
```bash
# Скопируйте ваши реальные изображения в папки:
cp /path/to/real/images/* data/real_test_cases/production/
cp /path/to/validation/images/* data/real_test_cases/validation/
```

### 2. Запуск инференса на реальных данных
```bash
# Для production кейсов
python scripts/test_real_cases.py --input production --model path/to/your/model.pt

# Для validation кейсов
python scripts/test_real_cases.py --input validation --model path/to/your/model.pt

# Пакетная обработка всех кейсов
python scripts/test_real_cases.py --input all --model path/to/your/model.pt
```

### 3. Анализ результатов
```bash
# Анализ результатов
python scripts/analyze_real_results.py --results-dir data/real_test_cases/results/

# Сравнение с Ground Truth (если доступен)
python scripts/compare_with_gt.py --results data/real_test_cases/results/ --gt data/annotations/
```

## Формат результатов

Результаты сохраняются в JSON формате:
```json
{
    "image_name": "case_001.jpg",
    "predictions": {
        "damage_level": 2,
        "confidence": 0.87,
        "damage_types": ["rust", "dent"],
        "bboxes": [...],
        "processing_time": 0.15
    },
    "metadata": {
        "model_version": "v1.0",
        "timestamp": "2025-09-13T10:30:00Z",
        "image_size": [1920, 1080]
    }
}
```
