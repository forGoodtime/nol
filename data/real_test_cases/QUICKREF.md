# Быстрая справка - Тестирование реальных случаев

## 🚗 Структура данных

```
data/real_test_cases/
├── production/          # Реальные случаи из продакшена
├── validation/          # Валидационные случаи  
├── results/            # Результаты инференса
├── analysis/           # Аналитические отчеты
├── gt_comparison/      # Сравнение с Ground Truth
├── metadata.json       # Метаданные о тестовых случаях
└── ground_truth_demo.json  # Пример Ground Truth
```

## 🛠 Быстрые команды

### Добавление ваших изображений
```bash
# Скопируйте изображения в соответствующие папки
cp /path/to/production/images/* data/real_test_cases/production/
cp /path/to/validation/images/* data/real_test_cases/validation/
```

### Тестирование
```bash
# Production случаи
python scripts/test_real_cases.py --input production

# Validation случаи
python scripts/test_real_cases.py --input validation

# Все случаи
python scripts/test_real_cases.py --input all

# С обученной моделью
python scripts/test_real_cases.py --input production --model path/to/your/model.pt

# Кастомная папка
python scripts/test_real_cases.py --custom-input /path/to/images --model path/to/model.pt
```

### Анализ результатов
```bash
# Полный анализ с графиками
python scripts/analyze_real_results.py

# Анализ кастомных результатов
python scripts/analyze_real_results.py --results-dir custom/results --output-dir custom/analysis
```

### Сравнение с Ground Truth
```bash
# Если у вас есть Ground Truth аннотации
python scripts/compare_with_gt.py \
    --results-dir data/real_test_cases/results \
    --gt-dir data/annotations

# С демо Ground Truth
python scripts/compare_with_gt.py \
    --results-dir data/real_test_cases/results \
    --gt-dir data/real_test_cases
```

## 📊 Результаты

После каждого тестирования вы получите:

- **JSON результаты**: детальные предсказания для каждого изображения
- **Аналитические графики**: распределения повреждений, уверенности, времени
- **Текстовые отчеты**: статистика и рекомендации
- **Сравнение с GT**: матрицы ошибок и метрики точности

## 🎯 Интерпретация

- **Damage Level**: 0=нет, 1=легкие, 2=средние, 3=серьезные повреждения
- **Confidence**: уверенность модели (0.0-1.0)
- **Processing Time**: время обработки изображения
- **Damage Types**: список типов повреждений

## 📖 Подробная документация

Полная инструкция в [QUICKSTART.md](../QUICKSTART.md) в разделе "Тестирование на реальных данных".
