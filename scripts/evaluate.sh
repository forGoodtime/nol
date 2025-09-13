#!/bin/bash
# Скрипт для оценки модели на тестовом наборе

set -e  # Останавливаться при ошибках

# Параметры по умолчанию
MODEL_PATH="experiments/best_model.pt"
CONFIG_PATH="src/config/default.yaml"
TEST_DATA_DIR="data/curated"
OUTPUT_DIR="experiments/evaluation"
BATCH_SIZE=32

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--data)
            TEST_DATA_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Использование: $0 [OPTIONS]"
            echo "OPTIONS:"
            echo "  -m, --model PATH       Путь к модели (по умолчанию: $MODEL_PATH)"
            echo "  -c, --config PATH      Путь к конфигу (по умолчанию: $CONFIG_PATH)"
            echo "  -d, --data PATH        Путь к данным (по умолчанию: $TEST_DATA_DIR)"
            echo "  -o, --output PATH      Папка для результатов (по умолчанию: $OUTPUT_DIR)"
            echo "  -b, --batch-size N     Размер батча (по умолчанию: $BATCH_SIZE)"
            echo "  -h, --help             Показать эту справку"
            exit 0
            ;;
        *)
            echo "Неизвестный параметр: $1"
            exit 1
            ;;
    esac
done

# Проверяем наличие файлов
if [ ! -f "$MODEL_PATH" ]; then
    echo "Ошибка: Модель не найдена по пути $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Ошибка: Конфиг не найден по пути $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Ошибка: Папка с данными не найдена: $TEST_DATA_DIR"
    exit 1
fi

# Создаем выходную папку
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ОЦЕНКА МОДЕЛИ ОБНАРУЖЕНИЯ ПОВРЕЖДЕНИЙ"
echo "=========================================="
echo "Модель: $MODEL_PATH"
echo "Конфигурация: $CONFIG_PATH"
echo "Данные: $TEST_DATA_DIR"
echo "Результаты: $OUTPUT_DIR"
echo "Размер батча: $BATCH_SIZE"
echo "=========================================="

# Запускаем инференс на тестовых данных
echo "🔍 Запуск инференса на тестовых данных..."
python src/inference/batch_infer.py \
    --input "$TEST_DATA_DIR" \
    --model "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR/predictions" \
    --batch-size "$BATCH_SIZE" \
    --save-visualizations

# Вычисляем метрики (если есть ground truth)
if [ -f "data/splits/test.txt" ] && [ -f "data/annotations/coco/instances.json" ]; then
    echo "📊 Вычисление метрик..."
    python scripts/compute_metrics.py \
        --predictions "$OUTPUT_DIR/predictions" \
        --ground-truth "data/annotations/coco/instances.json" \
        --test-split "data/splits/test.txt" \
        --output "$OUTPUT_DIR/metrics.json"
fi

# Генерируем отчет
echo "📝 Генерация отчета..."
python scripts/generate_report.py \
    --predictions-dir "$OUTPUT_DIR/predictions" \
    --metrics-file "$OUTPUT_DIR/metrics.json" \
    --output "$OUTPUT_DIR/evaluation_report.html"

echo "✅ Оценка завершена!"
echo "Результаты сохранены в: $OUTPUT_DIR"
echo "Отчет доступен: $OUTPUT_DIR/evaluation_report.html"

# Показываем краткую статистику
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo ""
    echo "📈 КРАТКАЯ СТАТИСТИКА:"
    python -c "
import json
with open('$OUTPUT_DIR/metrics.json', 'r') as f:
    metrics = json.load(f)
    
print(f\"  Общая точность: {metrics.get('accuracy', 'N/A'):.3f}\")
print(f\"  Средний F1-score: {metrics.get('f1_macro', 'N/A'):.3f}\")
print(f\"  Recall для критичных классов: {metrics.get('critical_recall', 'N/A'):.3f}\")
print(f\"  Обработано изображений: {metrics.get('total_images', 'N/A')}\")
"
fi
