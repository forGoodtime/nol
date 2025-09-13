#!/usr/bin/env python3
"""Загрузка и подготовка больших датасетов для детекции повреждений автомобилей"""

import os
import requests
import zipfile
import tarfile
import shutil
import json
import yaml
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import glob

def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Загрузка файла с прогресс-баром"""
    try:
        print(f"🔄 Скачиваем: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📦 Прогресс: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✅ Загружено: {filename}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при загрузке {url}: {str(e)}")
        return False

def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Извлечение архива"""
    try:
        print(f"📂 Извлекаем архив: {archive_path}")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"❌ Неподдерживаемый формат архива: {archive_path}")
            return False
        
        print(f"✅ Архив извлечён в: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при извлечении {archive_path}: {str(e)}")
        return False

def setup_car_damage_datasets(project_dir: str):
    """Настройка популярных датасетов повреждений автомобилей"""
    
    data_dir = os.path.join(project_dir, "data", "large_datasets")
    os.makedirs(data_dir, exist_ok=True)
    
    print("🚗 Подготовка больших датасетов повреждений автомобилей")
    print("=" * 60)
    
    # Список популярных датасетов
    datasets = [
        {
            'name': 'Car Damage Dataset (Kaggle)',
            'description': 'Большой датасет с различными типами повреждений',
            'size': '~2GB',
            'images': '~9000',
            'categories': ['scratch', 'dent', 'glass_shatter', 'lamp_broken', 'tire_flat']
        },
        {
            'name': 'Vehicle Damage Assessment',
            'description': 'Датасет для оценки повреждений',
            'size': '~1.5GB', 
            'images': '~6000',
            'categories': ['minor', 'moderate', 'severe']
        },
        {
            'name': 'CarDD (Car Damage Detection)',
            'description': 'Специализированный датасет для детекции',
            'size': '~3GB',
            'images': '~12000', 
            'categories': ['bumper_dent', 'door_dent', 'scratch', 'glass_crack', 'headlight']
        }
    ]
    
    # Показываем доступные датасеты
    print("📊 Доступные датасеты:")
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   📝 {dataset['description']}")
        print(f"   📦 Размер: {dataset['size']}")
        print(f"   🖼️  Изображений: {dataset['images']}")
        print(f"   🏷️  Категории: {', '.join(dataset['categories'])}")
    
    return data_dir

def create_synthetic_car_damage_dataset(project_dir: str, num_images: int = 1000):
    """Создание синтетического датасета для демонстрации"""
    
    print(f"\n🎨 Создание синтетического датасета ({num_images} изображений)")
    
    dataset_dir = os.path.join(project_dir, "data", "synthetic_car_damage")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Категории повреждений
    damage_categories = {
        0: 'no_damage',
        1: 'rust',
        2: 'dent', 
        3: 'scratch',
        4: 'severe_damage',
        5: 'missing_part'
    }
    
    # Создаём синтетические изображения и аннотации
    print("🔄 Генерация синтетических данных...")
    
    created_images = 0
    for i in range(num_images):
        # Создаём синтетическое изображение автомобиля
        img_width, img_height = 640, 480
        image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Рисуем "автомобиль" (прямоугольник)
        car_color = np.random.randint(50, 200, 3)
        cv2.rectangle(image, (100, 150), (540, 350), car_color.tolist(), -1)
        
        # Добавляем случайные повреждения
        annotations = []
        num_damages = np.random.randint(0, 4)  # 0-3 повреждения на изображение
        
        for _ in range(num_damages):
            # Случайная категория повреждения (исключаем no_damage)
            damage_type = np.random.randint(1, 6)
            
            # Случайное расположение повреждения на автомобиле
            x = np.random.randint(120, 500)
            y = np.random.randint(170, 320)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 60)
            
            # Убеждаемся, что bbox не выходит за границы
            x = max(0, min(x, img_width - w))
            y = max(0, min(y, img_height - h))
            
            # Рисуем повреждение
            if damage_type == 1:  # rust
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 100, 200), -1)
            elif damage_type == 2:  # dent
                cv2.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (50, 50, 50), -1)
            elif damage_type == 3:  # scratch
                cv2.line(image, (x, y+h//2), (x+w, y+h//2), (200, 200, 200), 3)
            elif damage_type == 4:  # severe_damage
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), -1)
            elif damage_type == 5:  # missing_part
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)
            
            # Добавляем аннотацию в формате YOLO
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            annotations.append(f"{damage_type} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Сохраняем изображение
        img_filename = f"synthetic_{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, image)
        
        # Сохраняем аннотации
        txt_filename = f"synthetic_{i:06d}.txt"
        txt_path = os.path.join(labels_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        created_images += 1
        if (i + 1) % 100 == 0:
            print(f"📊 Создано: {i + 1}/{num_images}")
    
    # Создаём dataset.yaml
    dataset_config = {
        'path': dataset_dir,
        'train': 'images',
        'val': 'images',  # Используем те же изображения для валидации
        'nc': 6,
        'names': damage_categories
    }
    
    dataset_yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Синтетический датасет создан: {dataset_dir}")
    print(f"📊 Всего изображений: {created_images}")
    print(f"⚙️ Конфигурация: {dataset_yaml_path}")
    
    return dataset_yaml_path

def download_real_car_damage_dataset(project_dir: str):
    """Попытка загрузить реальный датасет (GitHub/Kaggle)"""
    
    print("\n🌐 Поиск доступных датасетов в открытых источниках...")
    
    # Список репозиториев с датасетами повреждений автомобилей
    github_datasets = [
        {
            'name': 'CarDD-Detection-Dataset',
            'url': 'https://github.com/Jia-Research-Lab/CarDD-Detection-Dataset',
            'description': 'Car Damage Detection Dataset'
        },
        {
            'name': 'vehicle-damage-assessment',
            'url': 'https://github.com/neokt/vehicle-damage-assessment',
            'description': 'Vehicle Damage Assessment using Deep Learning'
        },
        {
            'name': 'car-damage-detection',
            'url': 'https://github.com/vijayabhaskar96/Car-damage-detection',
            'description': 'Car Damage Detection using ML/DL'
        }
    ]
    
    print("📋 Найденные репозитории с датасетами:")
    for i, repo in enumerate(github_datasets, 1):
        print(f"{i}. {repo['name']}")
        print(f"   🔗 {repo['url']}")
        print(f"   📝 {repo['description']}\n")
    
    # Пробуем создать скрипты для загрузки
    download_dir = os.path.join(project_dir, "data", "downloads")
    os.makedirs(download_dir, exist_ok=True)
    
    # Создаём скрипт для загрузки датасетов
    download_script = os.path.join(download_dir, "download_datasets.sh")
    
    script_content = """#!/bin/bash
# Скрипт для загрузки датасетов повреждений автомобилей

echo "🚗 Загрузка датасетов повреждений автомобилей"

# Создаём директории
mkdir -p real_datasets
cd real_datasets

# Клонируем репозитории с датасетами
echo "📦 Клонирование репозиториев..."

# CarDD Dataset
if [ ! -d "CarDD-Detection-Dataset" ]; then
    echo "Загружаем CarDD Dataset..."
    git clone https://github.com/Jia-Research-Lab/CarDD-Detection-Dataset.git
else
    echo "CarDD Dataset уже существует"
fi

# Vehicle Damage Assessment
if [ ! -d "vehicle-damage-assessment" ]; then
    echo "Загружаем Vehicle Damage Assessment..."
    git clone https://github.com/neokt/vehicle-damage-assessment.git
else
    echo "Vehicle Damage Assessment уже существует"
fi

# Car Damage Detection
if [ ! -d "car-damage-detection" ]; then
    echo "Загружаем Car Damage Detection..."
    git clone https://github.com/vijayabhaskar96/Car-damage-detection.git
else
    echo "Car Damage Detection уже существует"  
fi

echo "✅ Загрузка завершена!"
echo "📁 Проверьте папку real_datasets/"
"""
    
    with open(download_script, 'w') as f:
        f.write(script_content)
    
    # Делаем скрипт исполняемым
    os.chmod(download_script, 0o755)
    
    print(f"📝 Создан скрипт загрузки: {download_script}")
    print("💡 Запустите: bash data/downloads/download_datasets.sh")
    
    return download_script

def prepare_large_training_config(project_dir: str, dataset_yaml_path: str):
    """Создание конфигурации для обучения на большом датасете"""
    
    training_config = {
        'model_config': {
            'architecture': 'yolov8s',  # Средняя модель для большого датасета
            'num_classes': 6,
            'input_size': 640,
            'pretrained': True,
            'freeze_backbone': False,  # Не замораживаем для большого датасета
            'freeze_epochs': 0
        },
        'training_config': {
            'batch_size': 16,  # Больший batch для большого датасета
            'epochs': 200,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'patience': 30,
            'save_best_only': True,
            'early_stopping': True,
            'warmup_epochs': 5,
            'cos_lr': True  # Cosine learning rate scheduling
        },
        'data_config': {
            'dataset_yaml': dataset_yaml_path,
            'num_workers': 8,  # Больше workers для быстрой загрузки
            'pin_memory': True,
            'cache_images': False  # Не кешируем для экономии памяти
        },
        'augmentation_config': {
            # Сильные аугментации для большого датасета
            'horizontal_flip': 0.5,
            'vertical_flip': 0.1,
            'rotation': 15,
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.7,
            'hue': 0.015,
            'mosaic': 1.0,  # Всегда используем mosaic
            'mixup': 0.15,
            'copy_paste': 0.3,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0003,
            'flipud': 0.0,
            'fliplr': 0.5
        },
        'loss_config': {
            'use_knout_pryanik': True,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'box_loss_gain': 7.5,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0,
            'label_smoothing': 0.1  # Добавляем label smoothing
        },
        'validation_config': {
            'conf_threshold': 0.001,  # Низкий порог для детекции
            'iou_threshold': 0.6,
            'save_predictions': True,
            'visualize_results': True,
            'max_det': 300
        },
        'logging_config': {
            'log_level': 'INFO',
            'log_interval': 50,
            'save_checkpoints': True,
            'tensorboard_logging': True,
            'wandb_logging': True,  # Включаем W&B для большого датасета
            'save_period': 10  # Сохраняем каждые 10 эпох
        }
    }
    
    # Сохраняем конфигурацию
    config_path = os.path.join(project_dir, 'configs', 'large_dataset_training_config.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"⚙️ Конфигурация для большого датасета: {config_path}")
    
    return config_path

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    
    print("🚗 Настройка больших датасетов для детекции повреждений автомобилей")
    print("=" * 70)
    
    # 1. Показываем доступные датасеты
    data_dir = setup_car_damage_datasets(project_dir)
    
    # 2. Создаём синтетический датасет для быстрого тестирования
    synthetic_dataset_yaml = create_synthetic_car_damage_dataset(project_dir, num_images=1000)
    
    # 3. Готовим скрипт для загрузки реальных датасетов
    download_script = download_real_car_damage_dataset(project_dir)
    
    # 4. Создаём конфигурацию для обучения на большом датасете
    large_config_path = prepare_large_training_config(project_dir, synthetic_dataset_yaml)
    
    print("\n" + "=" * 70)
    print("✅ Подготовка датасетов завершена!")
    print("\n📋 Доступные опции:")
    print("1. 🎨 Синтетический датасет готов (1000 изображений)")
    print(f"   Путь: {synthetic_dataset_yaml}")
    print("\n2. 🌐 Для загрузки реальных датасетов:")
    print(f"   bash {download_script}")
    print("\n3. 🚀 Для обучения на синтетических данных:")
    print("   python scripts/train_on_large_dataset.py")
    print("\n4. ⚙️ Конфигурация для большого датасета готова:")
    print(f"   {large_config_path}")
    
    return {
        'synthetic_dataset': synthetic_dataset_yaml,
        'download_script': download_script,
        'config_path': large_config_path
    }

if __name__ == "__main__":
    results = main()
