#!/usr/bin/env python3
"""Тестирование обученной модели на большом датасете"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_trained_model(model_path: str, test_images_dir: str, output_dir: str, confidence_threshold: float = 0.01):
    """Тестирование обученной модели"""
    
    print(f"🧪 Тестирование модели: {model_path}")
    print(f"📁 Директория тестов: {test_images_dir}")
    print(f"🎯 Порог уверенности: {confidence_threshold}")
    
    # Создаём выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем модель
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Получаем список изображений для тестирования
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    
    if os.path.isdir(test_images_dir):
        for ext in image_extensions:
            test_images.extend([f for f in os.listdir(test_images_dir) 
                              if f.lower().endswith(ext.lower())])
    else:
        print(f"❌ Директория не найдена: {test_images_dir}")
        return None
    
    if not test_images:
        print(f"❌ Изображения не найдены в {test_images_dir}")
        return None
    
    print(f"📊 Найдено {len(test_images)} изображений для тестирования")
    
    # Категории повреждений
    damage_categories = {
        0: 'no_damage',
        1: 'rust',
        2: 'dent',
        3: 'scratch',
        4: 'severe_damage',
        5: 'missing_part'
    }
    
    # Цвета для визуализации
    colors = {
        'no_damage': (0, 255, 0),      # Зелёный
        'rust': (0, 165, 255),         # Оранжевый
        'dent': (255, 0, 0),           # Синий
        'scratch': (255, 255, 0),      # Голубой
        'severe_damage': (0, 0, 255),  # Красный
        'missing_part': (128, 0, 128)  # Фиолетовый
    }
    
    # Результаты тестирования
    test_results = {
        'model_path': model_path,
        'test_timestamp': datetime.now().isoformat(),
        'confidence_threshold': confidence_threshold,
        'total_images': len(test_images),
        'results': []
    }
    
    detection_stats = {cat: 0 for cat in damage_categories.values()}
    
    print(f"\n🔍 Начинаем тестирование...")
    
    for i, image_name in enumerate(test_images):
        image_path = os.path.join(test_images_dir, image_name)
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Не удалось загрузить: {image_name}")
            continue
        
        # Запускаем детекцию
        results = model(image_path, conf=confidence_threshold, iou=0.45)
        
        detections = []
        
        # Обрабатываем результаты
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if cls in damage_categories:
                            category_name = damage_categories[cls]
                            detection_stats[category_name] += 1
                            
                            detection = {
                                'class': category_name,
                                'confidence': float(conf),
                                'bbox': [float(x) for x in box],
                                'class_id': int(cls)
                            }
                            detections.append(detection)
        
        # Сохраняем результат для этого изображения
        image_result = {
            'image': image_name,
            'detections': detections,
            'total_detections': len(detections)
        }
        test_results['results'].append(image_result)
        
        # Создаём визуализацию
        if detections:
            visualized_image = visualize_detections(image, detections, colors)
            
            # Сохраняем визуализированное изображение
            output_image_path = os.path.join(output_dir, f"detected_{image_name}")
            cv2.imwrite(output_image_path, visualized_image)
        
        # Прогресс
        if (i + 1) % 10 == 0 or i == len(test_images) - 1:
            print(f"📊 Обработано: {i + 1}/{len(test_images)}")
    
    # Статистика
    total_detections = sum(detection_stats.values())
    test_results['statistics'] = {
        'total_detections': total_detections,
        'detections_by_category': detection_stats,
        'images_with_detections': len([r for r in test_results['results'] if r['total_detections'] > 0])
    }
    
    print(f"\n📊 Результаты тестирования:")
    print(f"  Всего детекций: {total_detections}")
    print(f"  Изображений с детекциями: {test_results['statistics']['images_with_detections']}/{len(test_images)}")
    
    print(f"\n📈 Детекции по категориям:")
    for category, count in detection_stats.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Сохраняем результаты
    results_file = os.path.join(output_dir, f"test_results_{confidence_threshold}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены: {results_file}")
    
    # Создаём сводную визуализацию
    create_summary_visualization(test_results, output_dir)
    
    return test_results

def visualize_detections(image, detections, colors):
    """Визуализация детекций на изображении"""
    
    result_image = image.copy()
    
    for detection in detections:
        category = detection['class']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Координаты bbox
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Цвет для категории
        color = colors.get(category, (255, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Подпись
        label = f"{category} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Фон для текста
        cv2.rectangle(result_image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Текст
        cv2.putText(result_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def create_summary_visualization(test_results, output_dir):
    """Создание сводной визуализации результатов"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Количество детекций по категориям
    categories = list(test_results['statistics']['detections_by_category'].keys())
    counts = list(test_results['statistics']['detections_by_category'].values())
    
    bars1 = ax1.bar(categories, counts, color=['green', 'orange', 'blue', 'cyan', 'red', 'purple'])
    ax1.set_title('Детекции по категориям', fontsize=14)
    ax1.set_ylabel('Количество')
    ax1.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # График 2: Распределение уверенности
    all_confidences = []
    for result in test_results['results']:
        for detection in result['detections']:
            all_confidences.append(detection['confidence'])
    
    if all_confidences:
        ax2.hist(all_confidences, bins=20, alpha=0.7, color='skyblue')
        ax2.set_title('Распределение уверенности детекций', fontsize=14)
        ax2.set_xlabel('Уверенность')
        ax2.set_ylabel('Количество')
    else:
        ax2.text(0.5, 0.5, 'Нет детекций', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Распределение уверенности детекций', fontsize=14)
    
    # График 3: Количество детекций на изображение
    detections_per_image = [result['total_detections'] for result in test_results['results']]
    ax3.hist(detections_per_image, bins=max(1, max(detections_per_image)) if detections_per_image else 1, 
             alpha=0.7, color='lightgreen')
    ax3.set_title('Количество детекций на изображение', fontsize=14)
    ax3.set_xlabel('Детекций на изображение')
    ax3.set_ylabel('Количество изображений')
    
    # График 4: Статистическая таблица
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Параметр', 'Значение'],
        ['Всего изображений', test_results['total_images']],
        ['Всего детекций', test_results['statistics']['total_detections']],
        ['Изображений с детекциями', test_results['statistics']['images_with_detections']],
        ['Порог уверенности', test_results['confidence_threshold']],
        ['Среднее детекций/изображение', 
         f"{test_results['statistics']['total_detections'] / test_results['total_images']:.2f}"]
    ]
    
    table = ax4.table(cellText=table_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax4.set_title('Сводная статистика', fontsize=14)
    
    plt.tight_layout()
    
    # Сохраняем визуализацию
    summary_path = os.path.join(output_dir, 'test_summary_visualization.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Сводная визуализация: {summary_path}")

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    
    # Пути к моделям
    models_to_test = [
        {
            'name': 'Original Small Model (7 images)',
            'path': os.path.join(project_dir, 'results', 'yolo_training', 'damage_detection_v1', 'weights', 'best.pt')
        },
        {
            'name': 'Large Dataset Model (1000 images)',  
            'path': os.path.join(project_dir, 'results', 'large_dataset_training', 'car_damage_v2_1000imgs', 'weights', 'best.pt')
        }
    ]
    
    # Директории с тестовыми изображениями
    test_directories = [
        {
            'name': 'Curated Real Images',
            'path': os.path.join(project_dir, 'data', 'curated')
        },
        {
            'name': 'Synthetic Test Images',
            'path': os.path.join(project_dir, 'data', 'synthetic_car_damage', 'images')
        }
    ]
    
    print("🧪 Тестирование обученных моделей")
    print("=" * 50)
    
    # Тестируем разные пороги уверенности
    confidence_thresholds = [0.001, 0.01, 0.05, 0.1, 0.25]
    
    for model_info in models_to_test:
        if not os.path.exists(model_info['path']):
            print(f"⚠️  Модель не найдена: {model_info['name']} - {model_info['path']}")
            continue
        
        print(f"\n🤖 Тестируем модель: {model_info['name']}")
        
        for test_dir_info in test_directories:
            if not os.path.exists(test_dir_info['path']):
                print(f"⚠️  Директория не найдена: {test_dir_info['name']}")
                continue
            
            print(f"\n📁 Тестовые данные: {test_dir_info['name']}")
            
            # Тестируем с разными порогами уверенности
            for conf_thresh in confidence_thresholds:
                output_dir = os.path.join(
                    project_dir, 'results', 'model_testing', 'comprehensive',
                    model_info['name'].replace(' ', '_').replace('(', '').replace(')', ''),
                    test_dir_info['name'].replace(' ', '_'),
                    f"conf_{conf_thresh}"
                )
                
                print(f"\n🎯 Тестирование с порогом уверенности: {conf_thresh}")
                
                try:
                    results = test_trained_model(
                        model_info['path'], 
                        test_dir_info['path'], 
                        output_dir, 
                        conf_thresh
                    )
                    
                    if results:
                        total_detections = results['statistics']['total_detections']
                        if total_detections > 0:
                            print(f"✅ Найдено {total_detections} детекций")
                        else:
                            print(f"❌ Детекций не найдено")
                
                except Exception as e:
                    print(f"💥 Ошибка при тестировании: {str(e)}")

if __name__ == "__main__":
    main()
