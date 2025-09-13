#!/usr/bin/env python3
"""Улучшенное тестирование модели с разными порогами уверенности"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
from datetime import datetime

def test_model_with_different_thresholds(model_path: str, test_dir: str, output_dir: str):
    """Тестирование модели с разными порогами уверенности"""
    
    print(f"🔍 Тестирование модели с разными порогами: {model_path}")
    
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
    
    # Тестируем с разными порогами
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_by_threshold = {}
    
    for threshold in thresholds:
        print(f"\n🎯 Тестирование с порогом уверенности: {threshold}")
        results_by_threshold[threshold] = []
        
        for img_file in image_files:
            img_path = os.path.join(test_dir, img_file)
            
            # Загружаем изображение
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Предсказание
            results = model(img_path, conf=threshold, iou=0.45, verbose=False)
            
            # Обработка результатов
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
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
                            'confidence': conf,
                            'bbox': box.tolist()
                        })
            
            results_by_threshold[threshold].append({
                'image': img_file,
                'detections': detections,
                'total_detections': len(detections)
            })
            
            if len(detections) > 0:
                print(f"  📸 {img_file}: найдено {len(detections)} объектов")
                for det in detections:
                    print(f"    - {det['class']}: {det['confidence']:.3f}")
    
    # Создаём детальный анализ
    create_threshold_analysis(results_by_threshold, output_dir)
    
    # Визуализируем лучшие результаты
    best_threshold = find_best_threshold(results_by_threshold)
    print(f"\n🎯 Лучший порог: {best_threshold}")
    
    visualize_best_results(model, test_dir, best_threshold, output_dir)
    
    return results_by_threshold

def find_best_threshold(results_by_threshold):
    """Найти лучший порог на основе количества детекций"""
    
    best_threshold = 0.25
    max_detections = 0
    
    for threshold, results in results_by_threshold.items():
        total_detections = sum(r['total_detections'] for r in results)
        if total_detections > max_detections:
            max_detections = total_detections
            best_threshold = threshold
    
    # Если ничего не найдено, используем самый низкий порог
    if max_detections == 0:
        best_threshold = min(results_by_threshold.keys())
    
    return best_threshold

def create_threshold_analysis(results_by_threshold, output_dir):
    """Создание анализа по порогам"""
    
    # Подготавливаем данные для графика
    thresholds = sorted(results_by_threshold.keys())
    total_detections = []
    
    for threshold in thresholds:
        total = sum(r['total_detections'] for r in results_by_threshold[threshold])
        total_detections.append(total)
    
    # Создаём график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Количество детекций по порогам
    ax1.plot(thresholds, total_detections, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Порог уверенности')
    ax1.set_ylabel('Общее количество детекций')
    ax1.set_title('Зависимость количества детекций от порога')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем значения на точки
    for i, (threshold, count) in enumerate(zip(thresholds, total_detections)):
        ax1.annotate(f'{count}', (threshold, count), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # График 2: Детализация по изображениям
    images_with_detections = []
    for threshold in thresholds:
        count = sum(1 for r in results_by_threshold[threshold] if r['total_detections'] > 0)
        images_with_detections.append(count)
    
    ax2.bar(range(len(thresholds)), images_with_detections, 
           color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.set_xlabel('Порог уверенности')
    ax2.set_ylabel('Изображений с детекциями')
    ax2.set_title('Количество изображений с найденными объектами')
    ax2.set_xticks(range(len(thresholds)))
    ax2.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45)
    
    # Добавляем значения на столбцы
    for i, count in enumerate(images_with_detections):
        ax2.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    
    # Сохраняем анализ
    analysis_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Анализ по порогам сохранён: {analysis_path}")
    
    # Сохраняем детальные результаты
    detailed_results = {
        'thresholds_tested': thresholds,
        'total_detections_by_threshold': dict(zip(thresholds, total_detections)),
        'images_with_detections': dict(zip(thresholds, images_with_detections)),
        'detailed_results': results_by_threshold
    }
    
    results_path = os.path.join(output_dir, 'detailed_threshold_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Детальные результаты: {results_path}")

def visualize_best_results(model, test_dir, threshold, output_dir):
    """Визуализация результатов с лучшим порогом"""
    
    print(f"🎨 Создание визуализации с порогом {threshold}")
    
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Предсказание
        results = model(img_path, conf=threshold, iou=0.45, verbose=False)
        
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
        
        detections_count = 0
        
        # Рисуем детекции
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
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
                    color = colors.get(class_name, 'yellow')
                    
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Прямоугольник
                    rect = plt.Rectangle((x1, y1), w, h, 
                                       linewidth=3, 
                                       edgecolor=color, 
                                       facecolor='none')
                    ax.add_patch(rect)
                    
                    # Подпись
                    label = f"{class_name}\n{conf:.3f}"
                    ax.text(x1, y1-10, label,
                            fontsize=12,
                            color=color,
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor='white',
                                     edgecolor=color,
                                     alpha=0.9))
                    
                    detections_count += 1
        
        title = f"{img_file}\nПорог: {threshold}, Найдено: {detections_count} объектов"
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Сохраняем
        output_filename = f"best_prediction_{os.path.splitext(img_file)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Основная функция тестирования"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    test_dir = os.path.join(project_dir, "data", "curated")
    output_dir = os.path.join(project_dir, "results", "enhanced_testing")
    model_path = os.path.join(project_dir, "results", "yolo_training", "damage_detection_v1", "weights", "best.pt")
    
    print("🧪 Улучшенное тестирование обученной модели AIinDrive")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print("❌ Обученная модель не найдена!")
        return
    
    print(f"✅ Найдена модель: {model_path}")
    
    # Тестирование с разными порогами
    results = test_model_with_different_thresholds(model_path, test_dir, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Улучшенное тестирование завершено!")
    print(f"📁 Результаты сохранены в: {output_dir}")
    
    # Сводка
    print("\n📊 Сводка по порогам:")
    for threshold, threshold_results in results.items():
        total_dets = sum(r['total_detections'] for r in threshold_results)
        images_with_dets = sum(1 for r in threshold_results if r['total_detections'] > 0)
        print(f"  Порог {threshold:4.2f}: {total_dets:2d} детекций в {images_with_dets} изображениях")

if __name__ == "__main__":
    main()
