#!/usr/bin/env python3
"""Поиск и загрузка актуальных датасетов повреждений автомобилей"""

import os
import requests
import json
from pathlib import Path
import zipfile
import subprocess

def find_available_datasets():
    """Поиск доступных датасетов повреждений автомобилей"""
    
    print("🔍 Поиск актуальных датасетов повреждений автомобилей...")
    
    # Альтернативные источники датасетов
    alternative_sources = [
        {
            'name': 'COCO Car Damage Dataset',
            'source': 'Roboflow Universe',
            'url': 'https://universe.roboflow.com/search?q=car%20damage',
            'description': 'Датасеты в формате COCO от сообщества',
            'format': 'COCO JSON'
        },
        {
            'name': 'Car Damage Detection (YOLOv8)',
            'source': 'Roboflow',
            'url': 'https://universe.roboflow.com/roboflow-jvuqo/car-damage-segmentation',
            'description': 'Готовый датасет для YOLOv8',
            'format': 'YOLOv8'
        },
        {
            'name': 'Vehicle Damage Dataset',
            'source': 'Kaggle',
            'url': 'https://www.kaggle.com/datasets/anujms/vehicle-damage-dataset',
            'description': 'Датасет с Kaggle',
            'format': 'Images + Labels'
        },
        {
            'name': 'Car Accident Dataset',
            'source': 'GitHub',
            'url': 'https://github.com/topics/car-damage-detection',
            'description': 'Открытые репозитории с датасетами',
            'format': 'Various'
        }
    ]
    
    print("📋 Найденные источники датасетов:")
    for i, source in enumerate(alternative_sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   📍 Источник: {source['source']}")
        print(f"   🔗 URL: {source['url']}")
        print(f"   📝 {source['description']}")
        print(f"   📁 Формат: {source['format']}")
    
    return alternative_sources

def download_roboflow_dataset(project_dir: str):
    """Загрузка датасета с Roboflow (требует API key)"""
    
    print("\n🤖 Инструкции по загрузке с Roboflow:")
    print("1. Зайдите на https://universe.roboflow.com")
    print("2. Найдите датасет 'car damage' или 'vehicle damage'")
    print("3. Создайте бесплатный аккаунт")
    print("4. Скопируйте код для загрузки датасета")
    
    roboflow_script = """
# Пример кода для загрузки с Roboflow:
# pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
dataset = project.version(1).download("yolov8")
"""
    
    script_path = os.path.join(project_dir, "scripts", "download_roboflow.py")
    with open(script_path, 'w') as f:
        f.write(roboflow_script)
    
    print(f"📝 Создан шаблон скрипта: {script_path}")
    
    return script_path

def create_enhanced_synthetic_dataset(project_dir: str, num_images: int = 5000):
    """Создание улучшенного синтетического датасета"""
    
    print(f"\n🎨 Создание улучшенного синтетического датасета ({num_images} изображений)")
    
    import cv2
    import numpy as np
    import random
    import yaml
    
    dataset_dir = os.path.join(project_dir, "data", "enhanced_synthetic_car_damage")
    train_dir = os.path.join(dataset_dir, "images", "train")
    val_dir = os.path.join(dataset_dir, "images", "val")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    val_labels_dir = os.path.join(dataset_dir, "labels", "val")
    
    # Создаём директории
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Более реалистичные цвета автомобилей
    car_colors = [
        (255, 255, 255),  # Белый
        (0, 0, 0),        # Чёрный
        (128, 128, 128),  # Серый
        (255, 0, 0),      # Красный
        (0, 0, 255),      # Синий
        (0, 128, 0),      # Зелёный
        (255, 255, 0),    # Жёлтый
        (128, 0, 128),    # Фиолетовый
    ]
    
    # Типы повреждений с более реалистичными параметрами
    damage_types = {
        1: {  # rust
            'name': 'rust',
            'color_range': [(0, 100, 150), (50, 150, 200)],
            'size_range': (15, 60),
            'probability': 0.3
        },
        2: {  # dent
            'name': 'dent',
            'color_range': [(30, 30, 30), (80, 80, 80)],
            'size_range': (25, 80),
            'probability': 0.25
        },
        3: {  # scratch
            'name': 'scratch',
            'color_range': [(150, 150, 150), (220, 220, 220)],
            'size_range': (40, 120),
            'probability': 0.25
        },
        4: {  # severe_damage
            'name': 'severe_damage',
            'color_range': [(0, 0, 200), (100, 100, 255)],
            'size_range': (60, 150),
            'probability': 0.15
        },
        5: {  # missing_part
            'name': 'missing_part',
            'color_range': [(0, 0, 0), (50, 50, 50)],
            'size_range': (30, 100),
            'probability': 0.1
        }
    }
    
    print("🔄 Генерация улучшенных синтетических данных...")
    
    # Разбиение train/val
    train_count = int(num_images * 0.8)
    val_count = num_images - train_count
    
    for i in range(num_images):
        # Определяем, куда сохранять (train или val)
        if i < train_count:
            img_dir = train_dir
            labels_dir = train_labels_dir
            prefix = "train"
        else:
            img_dir = val_dir
            labels_dir = val_labels_dir
            prefix = "val"
        
        # Создаём более реалистичное изображение
        img_width, img_height = 640, 480
        
        # Создаём фон (дорога, парковка)
        background_color = random.choice([
            (100, 100, 100),  # Асфальт
            (150, 150, 150),  # Бетон
            (120, 120, 120),  # Грязь
        ])
        image = np.full((img_height, img_width, 3), background_color, dtype=np.uint8)
        
        # Добавляем шум к фону
        noise = np.random.normal(0, 20, (img_height, img_width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Рисуем автомобиль более реалистично
        car_color = random.choice(car_colors)
        
        # Основной корпус автомобиля
        car_rect = (80, 120, 480, 240)  # x, y, width, height
        cv2.rectangle(image, (car_rect[0], car_rect[1]), 
                     (car_rect[0] + car_rect[2], car_rect[1] + car_rect[3]), 
                     car_color, -1)
        
        # Окна
        cv2.rectangle(image, (120, 140), (200, 200), (200, 200, 255), -1)  # Переднее
        cv2.rectangle(image, (420, 140), (500, 200), (200, 200, 255), -1)  # Заднее
        
        # Колёса
        cv2.circle(image, (140, 300), 30, (0, 0, 0), -1)     # Переднее левое
        cv2.circle(image, (500, 300), 30, (0, 0, 0), -1)     # Заднее левое
        
        # Фары
        cv2.circle(image, (70, 180), 15, (255, 255, 200), -1)  # Передняя
        cv2.circle(image, (570, 180), 15, (255, 100, 100), -1)  # Задняя
        
        annotations = []
        
        # Добавляем повреждения
        num_damages = np.random.poisson(1.5)  # Среднее количество повреждений
        num_damages = max(0, min(num_damages, 4))  # Ограничиваем 0-4
        
        for _ in range(num_damages):
            # Выбираем тип повреждения с учётом вероятности
            damage_probs = [damage_types[dt]['probability'] for dt in damage_types.keys()]
            damage_type = np.random.choice(list(damage_types.keys()), p=np.array(damage_probs)/sum(damage_probs))
            
            damage_info = damage_types[damage_type]
            
            # Случайное расположение повреждения на автомобиле
            margin = 20
            x = np.random.randint(car_rect[0] + margin, car_rect[0] + car_rect[2] - margin - 50)
            y = np.random.randint(car_rect[1] + margin, car_rect[1] + car_rect[3] - margin - 40)
            
            size_min, size_max = damage_info['size_range']
            w = np.random.randint(size_min, size_max)
            h = np.random.randint(size_min//2, size_max//2)
            
            # Убеждаемся, что bbox не выходит за границы
            x = max(0, min(x, img_width - w))
            y = max(0, min(y, img_height - h))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Выбираем цвет повреждения
            color_min, color_max = damage_info['color_range']
            color = [
                np.random.randint(color_min[0], color_max[0]),
                np.random.randint(color_min[1], color_max[1]),
                np.random.randint(color_min[2], color_max[2])
            ]
            
            # Рисуем повреждение в зависимости от типа
            if damage_type == 1:  # rust - неправильные пятна
                cv2.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 
                           np.random.randint(0, 360), 0, 360, color, -1)
                # Добавляем текстуру ржавчины
                for _ in range(random.randint(3, 8)):
                    px = x + random.randint(0, w)
                    py = y + random.randint(0, h)
                    cv2.circle(image, (px, py), random.randint(2, 5), color, -1)
                    
            elif damage_type == 2:  # dent - градиентные круги
                cv2.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, color, -1)
                # Добавляем блик для эффекта вмятины
                lighter_color = [min(255, c + 50) for c in color]
                cv2.ellipse(image, (x+w//3, y+h//3), (w//4, h//4), 0, 0, 360, lighter_color, -1)
                
            elif damage_type == 3:  # scratch - линии
                thickness = random.randint(2, 5)
                if random.random() > 0.5:  # Горизонтальная царапина
                    cv2.line(image, (x, y+h//2), (x+w, y+h//2), color, thickness)
                else:  # Вертикальная царапина
                    cv2.line(image, (x+w//2, y), (x+w//2, y+h), color, thickness)
                    
            elif damage_type == 4:  # severe_damage - крупные неправильные области
                # Создаём неправильную форму
                points = []
                for _ in range(random.randint(5, 8)):
                    px = x + random.randint(0, w)
                    py = y + random.randint(0, h)
                    points.append([px, py])
                cv2.fillPoly(image, [np.array(points)], color)
                
            elif damage_type == 5:  # missing_part - чёрные области
                cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
                # Добавляем границы для эффекта отсутствующей детали
                cv2.rectangle(image, (x, y), (x+w, y+h), (100, 100, 100), 2)
            
            # Добавляем аннотацию в формате YOLO
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            annotations.append(f"{damage_type} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Сохраняем изображение
        img_filename = f"{prefix}_{i:06d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, image)
        
        # Сохраняем аннотации
        txt_filename = f"{prefix}_{i:06d}.txt"
        txt_path = os.path.join(labels_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        if (i + 1) % 500 == 0:
            print(f"📊 Создано: {i + 1}/{num_images}")
    
    # Создаём dataset.yaml
    dataset_config = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 6,
        'names': {
            0: 'no_damage',
            1: 'rust',
            2: 'dent',
            3: 'scratch', 
            4: 'severe_damage',
            5: 'missing_part'
        }
    }
    
    dataset_yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Улучшенный синтетический датасет создан: {dataset_dir}")
    print(f"📊 Train: {train_count}, Val: {val_count}")
    print(f"⚙️ Конфигурация: {dataset_yaml_path}")
    
    return dataset_yaml_path

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    
    print("🚗 Поиск и подготовка датасетов повреждений автомобилей")
    print("=" * 70)
    
    # 1. Поиск доступных источников
    available_sources = find_available_datasets()
    
    # 2. Создание инструкций для загрузки с Roboflow
    roboflow_script = download_roboflow_dataset(project_dir)
    
    # 3. Создание улучшенного синтетического датасета
    enhanced_dataset_yaml = create_enhanced_synthetic_dataset(project_dir, num_images=5000)
    
    print("\n" + "=" * 70)
    print("✅ Поиск и подготовка датасетов завершены!")
    print("\n📋 Доступные опции:")
    print("1. 🎨 Улучшенный синтетический датасет готов (5000 изображений)")
    print(f"   Путь: {enhanced_dataset_yaml}")
    
    print("\n2. 🤖 Для загрузки реальных датасетов с Roboflow:")
    print("   - Зайдите на https://universe.roboflow.com")
    print("   - Найдите 'car damage detection' датасет")
    print(f"   - Используйте шаблон: {roboflow_script}")
    
    print("\n3. 📊 Другие источники:")
    print("   - Kaggle: https://www.kaggle.com/search?q=car+damage")
    print("   - GitHub: https://github.com/topics/car-damage")
    
    print("\n4. 🚀 Для обучения на улучшенном датасете:")
    print("   python scripts/train_on_enhanced_dataset.py")
    
    return enhanced_dataset_yaml

if __name__ == "__main__":
    enhanced_dataset_path = main()
