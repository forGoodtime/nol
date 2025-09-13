# 🚗 AIinDrive - Система обнаружения повреждений автомобилей

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-MVP-yellow" alt="Status">
</p>

Система автоматического обнаружения и анализа повреждений автомобилей с использованием компьютерного зрения и глубокого обучения. MVP версия для команды из 2 ML-инженеров.

## ✨ Основные возможности

- 🎯 **Обнаружение автомобилей** - детекция транспортных средств на изображениях
- 🔍 **Анализ повреждений** - сегментация и классификация дефектов:
  - Ржавчина и коррозия
  - Вмятины и царапины  
  - Отсутствие деталей
  - Загрязнения
- 🌡️ **Heatmap визуализация** - тепловые карты повреждений
- 📊 **Классификация серьезности** - 4 уровня повреждений (0-3)
- 🎯 **"Кнут и пряник"** - адаптивная система обучения
- 🌓 **Борьба с тенями/бликами** - специальная предобработка

## 📁 Структура проекта

```
AIinDrive/
├── 📊 data/                    # Данные и аннотации
│   ├── raw/                    # Исходные изображения
│   ├── curated/               # Отобранные данные
│   ├── annotations/           # COCO аннотации и маски
│   └── splits/                # Train/val/test разделения
├── 🧠 src/                     # Исходный код
│   ├── config/                # Конфигурации экспериментов
│   ├── datasets/              # DataLoader и аугментации
│   ├── models/                # Архитектуры моделей
│   ├── losses/                # Кастомные loss функции
│   ├── trainers/              # Циклы обучения
│   ├── utils/                 # Утилиты (тени, визуализация)
│   └── inference/             # Инференс и экспорт
├── 🔬 experiments/            # Логи экспериментов
├── 📓 notebooks/              # Jupyter notebooks (EDA)
├── 🛠️ scripts/                # Вспомогательные скрипты
├── 🐳 docker/                 # Docker контейнеры
└── 🧪 tests/                  # Тесты
```

## 🚀 Быстрый старт

### 1️⃣ Установка

```bash
# Клонирование репозитория
git clone <repo-url>
cd AIinDrive

# Создание окружения (выберите один вариант)
conda env create -f environment.yml && conda activate aiindrive
# или
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Установка проекта
pip install -e .
```

### 2️⃣ Демо запуск

```bash
# Создание демо данных и тестовый запуск
python demo.py --num-images 5

# Результат: созданы demo_data/ и demo_results/
```

### 3️⃣ Работа с реальными данными

```bash
# Подготовка данных
python scripts/prepare_data.py --action organize --source-dir /path/to/images

# EDA анализ
jupyter lab notebooks/eda.ipynb

# Обучение модели
python src/trainers/trainer_classification.py --config src/config/train_small.yaml

# Инференс
python src/inference/infer.py --input image.jpg --model model.pt --output results/
```

## 🏗️ Архитектура системы

### Основные компоненты:

1. **🔧 Предобработка**
   - Удаление теней и бликов (CLAHE, морфология)
   - Коррекция баланса белого
   - Адаптивное улучшение контраста

2. **🧠 Модели**
   - **Backbone**: ResNet, EfficientNet, MobileNet
   - **Сегментация**: U-Net, Mask R-CNN, YOLOv8-seg
   - **Anomaly Detection**: PatchCore, PaDiM
   - **Классификация**: Fusion head (RGB + heatmap → степень повреждения)

3. **📈 Loss функции**
   - **"Кнут и пряник"**: адаптивное обучение с поощрением/наказанием
   - **Взвешенный CrossEntropy**: балансировка классов
   - **Multi-task**: λ_seg * L_seg + λ_cls * L_cls + λ_kp * L_kp

### Классификация повреждений:

| Уровень | Описание | Примеры | Приоритет |
|---------|----------|---------|-----------|
| **0** | Нет повреждений | Чистый автомобиль | Низкий |
| **1** | Легкие повреждения | Царапины, грязь | Средний |
| **2** | Умеренные повреждения | Вмятины, коррозия | Высокий |
| **3** | Серьезные повреждения | Отсутствие деталей | **Критический** |

## 🎯 Механизм "Кнут и пряник"

Инновационная система адаптивного обучения:

```python
# Пряник: снижение loss при правильном предсказании  
if prediction == target:
    loss = base_loss * (1.0 - reward_factor)

# Кнут: увеличение loss при ошибке
else:
    loss = base_loss * (1.0 + penalty_factor)
    
# Усиленный кнут для критичных классов (уровень 3)
if target_class == 3:  # серьезные повреждения
    loss = loss * critical_multiplier
```

## 📊 Метрики и оценка

### Ключевые метрики:
- **Классификация**: Precision, Recall, F1 по классам
- **Сегментация**: IoU, Dice coefficient  
- **Калибровка**: ECE (Expected Calibration Error)
- **Критичные классы**: приоритет Recall > 95%

### Оценка модели:
```bash
# Полная оценка с отчетом
./scripts/evaluate.sh --model model.pt --data test_data/

# Получение метрик
python scripts/compute_metrics.py --predictions results/ --ground-truth annotations.json
```

## 🐳 Docker

### Обучение:
```bash
docker build -f docker/Dockerfile.train -t aiindrive:train .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/experiments:/app/experiments aiindrive:train
```

### Инференс:
```bash
docker build -f docker/Dockerfile.infer -t aiindrive:infer .
docker run -v $(pwd)/models:/app/models aiindrive:infer python src/inference/infer.py --input image.jpg
```

## 📈 Workflow обучения

1. **🔍 EDA** → Анализ данных, баланс классов, качество изображений
2. **⚖️ Baseline** → Быстрая классификация 0-3 без сегментации  
3. **🎯 Сегментация** → U-Net/Mask R-CNN или PatchCore/PaDiM
4. **🔗 Интеграция** → Fusion head: RGB + heatmap → final logits
5. **🎭 Кнут и пряник** → Кастомный loss с адаптацией
6. **🧪 Тестирование** → Стресс-тесты с тенями/ночью/углами
7. **🎯 Active Learning** → Hard negative mining

## 🔧 Конфигурация

Основные настройки в `src/config/default.yaml`:

```yaml
# Модель
model:
  backbone: "resnet50"
  num_classes: 4
  segmentation:
    enabled: true
    architecture: "unet"

# Кнут и пряник  
loss:
  knout_pryanik:
    enabled: true
    reward_factor: 0.1    # пряник
    penalty_factor: 0.3   # кнут
    adaptive: true

# Борьба с тенями
augmentation:
  clahe: true
  random_shadow: 0.3
  random_brightness: 0.4
```

## 📋 API Результатов

### JSON выход:
```json
{
  "image_path": "car_damage.jpg",
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
  },
  "segmentation": {
    "has_mask": true,
    "damage_areas": [
      {"type": "rust", "area": 1250, "bbox": [100, 150, 80, 60]},
      {"type": "dent", "area": 890, "bbox": [300, 200, 45, 55]}
    ]
  }
}
```

### Визуализации:
- `overlay.png` - изображение с наложенной heatmap
- `heatmap.png` - цветная тепловая карта
- `damage_info.png` - аннотированное изображение

## 🎯 MVP Цели

✅ **Выполнено:**
- Архитектура проекта
- Конфигурационная система
- Модули предобработки
- Loss функция "кнут и пряник"
- Инференс pipeline
- Docker контейнеры
- EDA notebook
- Демо система

🔄 **В процессе:**
- Обучение baseline модели  
- Интеграция сегментации
- Тестирование на реальных данных

🎯 **Планы:**
- WebUI интерфейс
- API сервер
- Мобильное приложение
- Интеграция с базами данных

## 🤝 Команда

Проект рассчитан на команду из **2 ML-инженеров (PyTorch)** + фронтенд/бэк разработчики.

### Роли:
- **ML Engineer #1**: Классификация, loss функции, метрики
- **ML Engineer #2**: Сегментация, anomaly detection, предобработка  
- **Frontend**: Веб-интерфейс, визуализация
- **Backend**: API, база данных, деплой

## 📚 Документация

- 📖 [Быстрый старт](QUICKSTART.md) - подробное руководство
- 📓 [EDA Notebook](notebooks/eda.ipynb) - анализ данных
- 🔧 [Конфигурации](src/config/) - настройки экспериментов
- 🐳 [Docker](docker/) - контейнеризация
- 📊 [Скрипты](scripts/) - вспомогательные утилиты

## 🐛 Troubleshooting

### Частые проблемы:

**CUDA не найден:**
```bash
python src/inference/infer.py --device cpu --input image.jpg --model model.pt
```

**Нехватка памяти:**
```bash
# Используйте train_small.yaml конфигурацию
python src/trainers/trainer_classification.py --config src/config/train_small.yaml
```

**Зависимости:**
```bash
pip install --force-reinstall -r requirements.txt
```

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 🎉 Заключение

Эта система обеспечивает:
- ✅ **Воспроизводимость** - Docker, конфигурации, фиксированные seeds
- ✅ **Масштабируемость** - модульная архитектура, микросервисы
- ✅ **Качество** - метрики, валидация, тестирование  
- ✅ **Скорость разработки** - готовые компоненты, демо, документация

**Готово к хакатону!** 🚀

---

<p align="center">
  <b>Создано командой AIinDrive 🚗💨</b>
</p>
