# Быстрый старт AIinDrive

## Установка

### 1. Клонирование репозитория
```bash
git clone <repo-url>
cd AIinDrive
```

### 2. Создание окружения

#### Вариант A: conda (рекомендуется)
```bash
conda env create -f environment.yml
conda activate aiindrive
```

#### Вариант B: pip + venv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Установка проекта
```bash
pip install -e .
```

## Подготовка данных

### 1. Организация данных
```bash
# Копирование и организация изображений
python scripts/prepare_data.py --action organize --source-dir /path/to/your/images --target-dir .

# Создание примера аннотаций (для тестирования)
python scripts/prepare_data.py --action create_sample --images-dir data/curated

# Создание train/val/test разбиения
python scripts/prepare_data.py --action split \
    --images-dir data/curated \
    --annotations data/annotations/coco/instances.json \
    --output-dir data/splits
```

### 2. Валидация данных
```bash
python scripts/prepare_data.py --action validate \
    --images-dir data/curated \
    --annotations data/annotations/coco/instances.json
```

## Исследовательский анализ данных (EDA)

```bash
# Запуск Jupyter notebook
jupyter lab notebooks/eda.ipynb
```

## Обучение модели

### 1. Baseline обучение
```bash
# Быстрое обучение на малом датасете
python src/trainers/trainer_classification.py --config src/config/train_small.yaml

# Полное обучение
python src/trainers/trainer_classification.py --config src/config/default.yaml
```

### 2. Совместное обучение (сегментация + классификация)
```bash
python src/trainers/trainer_joint.py --config src/config/default.yaml
```

### 3. Мониторинг обучения
```bash
# Если используется wandb
wandb login
# Логи будут доступны на wandb.ai

# Если используется tensorboard
tensorboard --logdir experiments/
```

## Инференс

### 1. Одно изображение
```bash
python src/inference/infer.py \
    --input path/to/image.jpg \
    --model experiments/best_model.pt \
    --output results/
```

### 2. Батчевая обработка
```bash
python src/inference/infer.py \
    --input path/to/images/folder/ \
    --model experiments/best_model.pt \
    --output results/ \
    --batch
```

### 3. Отключение предобработки теней
```bash
python src/inference/infer.py \
    --input image.jpg \
    --model model.pt \
    --no-shadow-removal
```

## Оценка модели

```bash
# Полная оценка с генерацией отчета
./scripts/evaluate.sh --model experiments/best_model.pt --data data/curated

# Ручная оценка
python src/inference/batch_infer.py \
    --input data/curated \
    --model experiments/best_model.pt \
    --output evaluation_results
```

## Тестирование на реальных данных

AIinDrive поддерживает тестирование модели на реальных случаях без аннотаций для валидации в продакшене.

### 1. Подготовка реальных тестовых данных

```bash
# Создание структуры папок
mkdir -p data/real_test_cases/production
mkdir -p data/real_test_cases/validation

# Добавление ваших реальных изображений
cp /path/to/production/images/* data/real_test_cases/production/
cp /path/to/validation/images/* data/real_test_cases/validation/
```

### 2. Запуск тестирования

```bash
# Тестирование production кейсов
python scripts/test_real_cases.py --input production --model path/to/your/model.pt

# Тестирование validation кейсов  
python scripts/test_real_cases.py --input validation --model path/to/your/model.pt

# Тестирование всех реальных кейсов
python scripts/test_real_cases.py --input all --model path/to/your/model.pt

# Тестирование кастомной папки с изображениями
python scripts/test_real_cases.py --custom-input /path/to/custom/images --model path/to/your/model.pt
```

### 3. Анализ результатов

```bash
# Генерация аналитического отчета и графиков
python scripts/analyze_real_results.py

# Анализ результатов из кастомной папки
python scripts/analyze_real_results.py --results-dir data/real_test_cases/results --output-dir custom_analysis
```

### 4. Сравнение с Ground Truth (опционально)

Если у вас есть Ground Truth аннотации для реальных случаев:

```bash
# Сравнение результатов модели с Ground Truth
python scripts/compare_with_gt.py \
    --results-dir data/real_test_cases/results \
    --gt-dir data/annotations \
    --output-dir data/real_test_cases/gt_comparison
```

### 5. Структура результатов

После тестирования вы получите:
```
data/real_test_cases/
├── results/                    # Результаты инференса
│   ├── production_results/     # JSON файлы с предсказаниями 
│   └── validation_results/     # для каждого изображения
├── analysis/                   # Аналитические отчеты
│   ├── analysis_overview.png   # Графики распределений
│   ├── analysis_report.txt     # Текстовый отчет
│   └── detailed_stats.json     # Детальная статистика
└── gt_comparison/             # Сравнение с Ground Truth (опционально)
    ├── gt_comparison.png      # Матрица ошибок и метрики
    └── gt_comparison_report.txt
```

### 6. Интерпретация результатов

- **Damage Level**: 0 = нет повреждений, 1 = легкие, 2 = средние, 3 = серьезные
- **Confidence**: уверенность модели (0.0-1.0)
- **Processing Time**: время обработки одного изображения
- **Damage Types**: список обнаруженных типов повреждений

## Docker

### 1. Обучение в Docker
```bash
# Сборка образа
docker build -f docker/Dockerfile.train -t aiindrive:train .

# Запуск обучения
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/experiments:/app/experiments aiindrive:train
```

### 2. Инференс в Docker
```bash
# Сборка образа для инференса
docker build -f docker/Dockerfile.infer -t aiindrive:infer .

# Запуск инференса
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data aiindrive:infer \
    python src/inference/infer.py --input /app/data/test_image.jpg --model /app/models/model.pt
```

## Структура результатов

После выполнения инференса вы получите:

```
results/
├── image_name_results.json     # JSON с метриками и предсказаниями
├── image_name_overlay.png      # Изображение с наложенной heatmap
├── image_name_heatmap.png      # Цветная heatmap повреждений
└── image_name_damage_info.png  # Изображение с информацией о повреждениях
```

### Пример JSON результата:
```json
{
  "image_path": "path/to/image.jpg",
  "classification": {
    "predicted_class": 2,
    "class_name": "Moderate damage",
    "confidence": 0.857,
    "class_probabilities": {
      "No damage": 0.023,
      "Light damage": 0.120,
      "Moderate damage": 0.857,
      "Severe damage": 0.000
    }
  }
}
```

## Конфигурация

Основные параметры находятся в `src/config/default.yaml`. Ключевые секции:

- `model`: архитектура модели и backbone
- `training`: параметры обучения
- `loss.knout_pryanik`: настройки "кнута и пряника"
- `augmentation`: параметры аугментаций
- `data`: пути к данным и размеры

## Troubleshooting

### Проблемы с CUDA
```bash
# Проверка доступности CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Использование CPU
python src/inference/infer.py --device cpu --input image.jpg --model model.pt
```

### Проблемы с памятью
```bash
# Уменьшение размера батча
python src/trainers/trainer_classification.py --config src/config/train_small.yaml

# Уменьшение размера изображений в конфиге
# Измените data.image_size в .yaml файле
```

### Проблемы с зависимостями
```bash
# Переустановка зависимостей
pip install --force-reinstall -r requirements.txt

# Обновление pip
pip install --upgrade pip setuptools wheel
```

## Дополнительные возможности

### Экспорт модели
```bash
python src/inference/export_torchscript.py \
    --model experiments/best_model.pt \
    --output model.torchscript
```

### Создание demo интерфейса
```bash
# Gradio интерфейс (если установлен)
python scripts/demo_gradio.py --model experiments/best_model.pt
```

### Интеграция с API
```bash
# FastAPI сервер
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
