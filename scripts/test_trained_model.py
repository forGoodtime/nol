#!/usr/bin/env python3
"""Тестирование обученной модели"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
from datetime import datetime

def find_best_model(project_dir: str):
    """Поиск лучшей обученной модели"""
    
    results_dir = os.path.join(project_dir, "results", "yolo_training")
    
    # Ищем последнюю обученную модель
    possible_paths = [
        os.path.join(results_dir, "damage_detection_v1", "weights", "best.pt"),
        os.path.join(results_dir, "damage_detection_v1", "weights", "last.pt")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Если не нашли, ищем любую модель в results
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file in ["best.pt", "last.pt"]:
                return os.path.join(root, file)
    
    return None

def test_model_on_images(model_path: str, test_dir: str, output_dir: str):
    """Тестирование модели на изображениях"""
    
    print(f"🔍 Тестирование модели: {model_path}")
    
    # Загружаем модель
    model = YOLO(model_path)
    
    # Создаём выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем тестовые изображения
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ Тестовые изображения не найдены!")
        return
    
    print(f"📊 Найдено {len(image_files)} тестовых изображений")
    
    results_summary = {
        'model_path': model_path,
        'test_timestamp': datetime.now().isoformat(),
        'results': []
    }
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"🔍 Обрабатываем: {img_file}")
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Предсказание
        results = model(img_path, conf=0.25, iou=0.45)
        
        # Обработка результатов
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Имена классов
                    class_names = {
                        0: 'no_damage',
                        1: 'rust',
                        2: 'dent',
                        3: 'scratch', 
                        4: 'severe_damage',
                        5: 'missing_part'
                    }
                    
                    class_name = class_names.get(cls, f'class_{cls}')
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })
        
        # Сохраняем результат
        img_result = {
            'image': img_file,
            'detections': detections,
            'total_detections': len(detections)
        }
        results_summary['results'].append(img_result)
        
        # Визуализация
        visualize_predictions(image, detections, img_file, output_dir)
        
        print(f"  ✅ Найдено {len(detections)} объектов")
        for det in detections:
            print(f"    - {det['class']}: {det['confidence']:.3f}")
    
    # Сохраняем сводку результатов
    summary_path = os.path.join(output_dir, 'test_results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Сводка результатов: {summary_path}")
    
    return results_summary

def visualize_predictions(image, detections, filename, output_dir):
    """Визуализация предсказаний"""
    
    # Конвертируем BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Создаём фигуру
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Цвета для классов
    colors = {
        'no_damage': 'green',
        'rust': 'orange',
        'dent': 'blue',
        'scratch': 'cyan',
        'severe_damage': 'red',
        'missing_part': 'purple'
    }
    
    # Рисуем детекции
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1
        
        color = colors.get(det['class'], 'yellow')
        
        # Прямоугольник
        rect = plt.Rectangle((x1, y1), w, h, 
                           linewidth=2, 
                           edgecolor=color, 
                           facecolor='none')
        ax.add_patch(rect)
        
        # Подпись
        label = f"{det['class']}\n{det['confidence']:.3f}"
        ax.text(x1, y1-10, label,
                fontsize=10,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor='white',
                         edgecolor=color,
                         alpha=0.8))
    
    ax.set_title(f"Детекция повреждений: {filename}", fontsize=14)
    ax.axis('off')
    
    # Сохраняем
    output_filename = f"prediction_{os.path.splitext(filename)[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_test_report(results_summary, output_dir):
    """Генерация отчёта по тестированию"""
    
    print("📊 Генерация отчёта по тестированию...")
    
    total_images = len(results_summary['results'])
    total_detections = sum(r['total_detections'] for r in results_summary['results'])
    
    # Статистика по классам
    class_stats = {}
    for result in results_summary['results']:
        for det in result['detections']:
            cls = det['class']
            if cls not in class_stats:
                class_stats[cls] = {'count': 0, 'avg_conf': 0, 'confidences': []}
            class_stats[cls]['count'] += 1
            class_stats[cls]['confidences'].append(det['confidence'])
    
    # Вычисляем средние уверенности
    for cls, stats in class_stats.items():
        stats['avg_conf'] = np.mean(stats['confidences'])
        stats['max_conf'] = np.max(stats['confidences'])
        stats['min_conf'] = np.min(stats['confidences'])
    
    # Создаём графики
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Количество детекций по классам
    if class_stats:
        classes = list(class_stats.keys())
        counts = [class_stats[cls]['count'] for cls in classes]
        colors_list = ['green', 'orange', 'blue', 'cyan', 'red', 'purple']
        
        bars1 = ax1.bar(classes, counts, color=colors_list[:len(classes)])
        ax1.set_title('Количество детекций по классам', fontsize=14)
        ax1.set_ylabel('Количество')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
    
    # График 2: Средняя уверенность по классам
    if class_stats:
        avg_confs = [class_stats[cls]['avg_conf'] for cls in classes]
        bars2 = ax2.bar(classes, avg_confs, color=colors_list[:len(classes)])
        ax2.set_title('Средняя уверенность по классам', fontsize=14)
        ax2.set_ylabel('Уверенность')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        for bar, conf in zip(bars2, avg_confs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.3f}', ha='center', va='bottom')
    
    # График 3: Распределение уверенности
    if class_stats:
        all_confidences = []
        for stats in class_stats.values():
            all_confidences.extend(stats['confidences'])
        
        ax3.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Распределение уверенности детекций', fontsize=14)
        ax3.set_xlabel('Уверенность')
        ax3.set_ylabel('Количество')
    
    # График 4: Информационная таблица
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Метрика', 'Значение']]
    table_data.append(['Всего изображений', str(total_images)])
    table_data.append(['Всего детекций', str(total_detections)])
    table_data.append(['Среднее детекций на изображение', f'{total_detections/total_images:.2f}' if total_images > 0 else '0'])
    
    if class_stats:
        table_data.append(['Найдено классов', str(len(class_stats))])
        avg_all_conf = np.mean([stats['avg_conf'] for stats in class_stats.values()])
        table_data.append(['Средняя уверенность', f'{avg_all_conf:.3f}'])
    
    table = ax4.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax4.set_title('Сводная статистика', fontsize=14)
    
    plt.tight_layout()
    
    # Сохраняем отчёт
    report_path = os.path.join(output_dir, 'test_report.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Отчёт сохранён: {report_path}")

def main():
    """Основная функция тестирования"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    test_dir = os.path.join(project_dir, "data", "curated")
    output_dir = os.path.join(project_dir, "results", "model_testing")
    
    print("🧪 Тестирование обученной модели AIinDrive")
    print("=" * 60)
    
    # Поиск модели
    model_path = find_best_model(project_dir)
    
    if not model_path:
        print("❌ Обученная модель не найдена!")
        print("💡 Сначала запустите обучение: python scripts/simple_train.py")
        return
    
    print(f"✅ Найдена модель: {model_path}")
    
    # Тестирование
    results_summary = test_model_on_images(model_path, test_dir, output_dir)
    
    # Генерация отчёта
    generate_test_report(results_summary, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Тестирование завершено!")
    print(f"📁 Результаты сохранены в: {output_dir}")
    print("\n📊 Статистика:")
    print(f"  Тестовых изображений: {len(results_summary['results'])}")
    total_dets = sum(r['total_detections'] for r in results_summary['results'])
    print(f"  Всего детекций: {total_dets}")
    
    if results_summary['results']:
        avg_dets = total_dets / len(results_summary['results'])
        print(f"  Среднее детекций на изображение: {avg_dets:.2f}")

if __name__ == "__main__":
    main()
