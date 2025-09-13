#!/usr/bin/env python3
"""Визуализация аннотаций для проверки"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def visualize_coco_annotations(data_dir: str, annotation_file: str, output_dir: str):
    """Визуализация COCO аннотаций"""
    
    print("🎨 Создание визуализации аннотаций...")
    
    # Создаём выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем аннотации
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Создаём маппинги
    images_map = {img['id']: img for img in coco_data['images']}
    categories_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Группируем аннотации по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in annotations_by_image:
            annotations_by_image[ann['image_id']] = []
        annotations_by_image[ann['image_id']].append(ann)
    
    # Цвета для категорий
    colors = {
        'no_damage': (0, 255, 0),      # Зелёный
        'rust': (0, 165, 255),         # Оранжевый
        'dent': (255, 0, 0),           # Синий  
        'scratch': (255, 255, 0),      # Голубой
        'severe_damage': (0, 0, 255),  # Красный
        'missing_part': (128, 0, 128)  # Фиолетовый
    }
    
    # Обрабатываем каждое изображение
    for img_id, img_info in images_map.items():
        img_path = os.path.join(data_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"⚠️  Изображение не найдено: {img_path}")
            continue
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Не удалось загрузить: {img_path}")
            continue
        
        # Конвертируем BGR -> RGB для matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создаём фигуру
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Добавляем аннотации
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                x, y, w, h = ann['bbox']
                category_name = categories_map[ann['category_id']]
                color = colors.get(category_name, (255, 255, 255))
                
                # Нормализуем цвет для matplotlib (0-1)
                color_norm = tuple(c/255.0 for c in color)
                
                # Рисуем прямоугольник
                rect = Rectangle((x, y), w, h, 
                               linewidth=2, 
                               edgecolor=color_norm, 
                               facecolor='none')
                ax.add_patch(rect)
                
                # Добавляем подпись
                damage_level = ann.get('damage_level', 'N/A')
                confidence = ann.get('attributes', {}).get('confidence', 0.0)
                label = f"{category_name}\nLevel: {damage_level}\nConf: {confidence:.2f}"
                
                ax.text(x, y-10, label, 
                       fontsize=10, 
                       color=color_norm,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', 
                                edgecolor=color_norm,
                                alpha=0.8))
        
        # Настройки отображения
        ax.set_title(f"Изображение: {img_info['file_name']}", fontsize=14)
        ax.axis('off')
        
        # Сохраняем визуализацию
        output_filename = f"visualization_{os.path.splitext(img_info['file_name'])[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Сохранено: {output_path}")
    
    # Создаём сводную статистику
    create_annotation_summary(coco_data, output_dir)

def create_annotation_summary(coco_data, output_dir):
    """Создание сводной статистики по аннотациям"""
    
    # Статистика по категориям
    category_stats = {}
    damage_level_stats = {}
    
    for cat in coco_data['categories']:
        category_stats[cat['name']] = {'count': 0, 'total_area': 0}
    
    for ann in coco_data['annotations']:
        cat_name = next(cat['name'] for cat in coco_data['categories'] 
                       if cat['id'] == ann['category_id'])
        
        category_stats[cat_name]['count'] += 1
        category_stats[cat_name]['total_area'] += ann['area']
        
        # Статистика по уровням повреждений
        damage_level = ann.get('damage_level', 0)
        if damage_level not in damage_level_stats:
            damage_level_stats[damage_level] = 0
        damage_level_stats[damage_level] += 1
    
    # Создаём график статистики
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Количество аннотаций по категориям
    categories = list(category_stats.keys())
    counts = [category_stats[cat]['count'] for cat in categories]
    
    bars1 = ax1.bar(categories, counts, color=['green', 'orange', 'blue', 'cyan', 'red', 'purple'])
    ax1.set_title('Количество аннотаций по категориям', fontsize=14)
    ax1.set_ylabel('Количество')
    ax1.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # График 2: Общая площадь повреждений по категориям
    areas = [category_stats[cat]['total_area'] for cat in categories]
    bars2 = ax2.bar(categories, areas, color=['green', 'orange', 'blue', 'cyan', 'red', 'purple'])
    ax2.set_title('Общая площадь повреждений по категориям', fontsize=14)
    ax2.set_ylabel('Площадь (пиксели)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, area in zip(bars2, areas):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{int(area)}', ha='center', va='bottom')
    
    # График 3: Распределение по уровням повреждений
    levels = list(damage_level_stats.keys())
    level_counts = list(damage_level_stats.values())
    level_labels = [f'Уровень {level}' for level in levels]
    
    colors3 = ['green', 'yellow', 'orange', 'red'][:len(levels)]
    ax3.pie(level_counts, labels=level_labels, colors=colors3, autopct='%1.1f%%')
    ax3.set_title('Распределение по уровням повреждений', fontsize=14)
    
    # График 4: Информационная таблица
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Категория', 'Количество', 'Средняя площадь']]
    for cat, stats in category_stats.items():
        avg_area = stats['total_area'] / stats['count'] if stats['count'] > 0 else 0
        table_data.append([cat, stats['count'], f"{avg_area:.0f}"])
    
    table = ax4.table(cellText=table_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Детальная статистика', fontsize=14)
    
    # Сохраняем статистику
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'annotation_statistics.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Статистика сохранена: {summary_path}")

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    data_dir = os.path.join(project_dir, "data", "curated")
    annotation_file = os.path.join(project_dir, "data", "annotations", "coco", "instances.json")
    output_dir = os.path.join(project_dir, "visualization", "annotations")
    
    print("🎨 Визуализация аннотаций AIinDrive")
    print("=" * 50)
    
    if not os.path.exists(annotation_file):
        print(f"❌ Файл аннотаций не найден: {annotation_file}")
        print("💡 Запустите сначала: python scripts/prepare_training.py")
        return
    
    # Создаём визуализации
    visualize_coco_annotations(data_dir, annotation_file, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ Визуализация завершена!")
    print(f"📁 Результаты сохранены в: {output_dir}")
    print("\n🔍 Проверьте аннотации и при необходимости скорректируйте их")
    print("💡 Затем запустите обучение: python scripts/train.py")

if __name__ == "__main__":
    main()
