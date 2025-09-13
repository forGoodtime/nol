#!/usr/bin/env python3
"""Подготовка данных для обучения модели"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import cv2
import numpy as np

def validate_dataset(data_dir: str, annotation_file: str) -> Dict[str, any]:
    """Валидация датасета"""
    
    print("🔍 Проверка датасета...")
    
    # Загружаем аннотации
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Проверяем изображения
    images = {}
    missing_images = []
    valid_images = 0
    
    for img_info in coco_data['images']:
        img_path = os.path.join(data_dir, img_info['file_name'])
        if os.path.exists(img_path):
            # Проверяем, что изображение читается
            image = cv2.imread(img_path)
            if image is not None:
                height, width = image.shape[:2]
                images[img_info['id']] = {
                    'path': img_path,
                    'actual_size': (width, height),
                    'expected_size': (img_info['width'], img_info['height']),
                    'valid': True
                }
                valid_images += 1
            else:
                images[img_info['id']] = {'valid': False, 'error': 'Cannot read image'}
        else:
            missing_images.append(img_info['file_name'])
            images[img_info['id']] = {'valid': False, 'error': 'File not found'}
    
    # Статистика по категориям
    category_stats = {}
    for cat in coco_data['categories']:
        category_stats[cat['id']] = {
            'name': cat['name'],
            'count': 0
        }
    
    # Подсчёт аннотаций
    valid_annotations = 0
    for ann in coco_data['annotations']:
        if ann['image_id'] in images and images[ann['image_id']]['valid']:
            category_stats[ann['category_id']]['count'] += 1
            valid_annotations += 1
    
    # Результаты валидации
    validation_results = {
        'total_images': len(coco_data['images']),
        'valid_images': valid_images,
        'missing_images': missing_images,
        'total_annotations': len(coco_data['annotations']),
        'valid_annotations': valid_annotations,
        'category_distribution': category_stats,
        'images_info': images
    }
    
    # Отчёт
    print(f"✅ Изображений: {valid_images}/{len(coco_data['images'])}")
    print(f"✅ Аннотаций: {valid_annotations}/{len(coco_data['annotations'])}")
    
    if missing_images:
        print(f"❌ Отсутствующие изображения: {missing_images}")
    
    print("\n📊 Распределение по категориям:")
    for cat_id, stats in category_stats.items():
        print(f"  {stats['name']}: {stats['count']} аннотаций")
    
    return validation_results

def create_train_val_split(data_dir: str, train_ratio: float = 0.7) -> Tuple[List[str], List[str]]:
    """Создание разбиения train/val"""
    
    print(f"\n🔄 Создание разбиения train/val (ratio: {train_ratio})")
    
    # Получаем все изображения
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # Разбиение
    n_train = int(len(image_files) * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"📦 Train: {len(train_files)} изображений")
    print(f"📦 Val: {len(val_files)} изображений")
    
    for f in train_files:
        print(f"  Train: {f}")
    for f in val_files:
        print(f"  Val: {f}")
    
    return train_files, val_files

def create_yolo_annotations(coco_file: str, output_dir: str, image_dir: str):
    """Создание аннотаций в формате YOLO"""
    
    print(f"\n🔄 Создание YOLO аннотаций в {output_dir}")
    
    # Создаём директории
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем COCO данные
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Создаём маппинг изображений
    images_map = {img['id']: img for img in coco_data['images']}
    
    # Группируем аннотации по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] not in annotations_by_image:
            annotations_by_image[ann['image_id']] = []
        annotations_by_image[ann['image_id']].append(ann)
    
    # Создаём YOLO файлы
    for img_id, img_info in images_map.items():
        txt_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    # Конвертируем bbox в YOLO формат
                    x, y, w, h = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']
                    
                    # Нормализация
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    width = w / img_w
                    height = h / img_h
                    
                    # YOLO format: class x_center y_center width height
                    f.write(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✅ Создано {len(images_map)} YOLO аннотаций")

def create_training_config(project_dir: str, train_files: List[str], val_files: List[str]):
    """Создание конфигурации для обучения"""
    
    print(f"\n🔄 Создание конфигурации обучения")
    
    config = {
        'model_config': {
            'architecture': 'yolov8n',  # Лёгкая модель для маленького датасета
            'num_classes': 6,
            'input_size': 640,
            'pretrained': True
        },
        'training_config': {
            'batch_size': 4,  # Маленький batch для 7 изображений
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'patience': 20,
            'save_best_only': True,
            'early_stopping': True
        },
        'data_config': {
            'dataset_path': 'data/curated',
            'annotation_path': 'data/annotations/coco/instances.json',
            'train_files': train_files,
            'val_files': val_files,
            'num_workers': 2,
            'pin_memory': True
        },
        'augmentation_config': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.1,
            'rotation': 10,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'mosaic': 0.3,
            'mixup': 0.1
        },
        'loss_config': {
            'use_knout_pryanik': True,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'box_loss_gain': 7.5,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0
        },
        'validation_config': {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'save_predictions': True,
            'visualize_results': True
        },
        'logging_config': {
            'log_level': 'INFO',
            'log_interval': 10,
            'save_checkpoints': True,
            'tensorboard_logging': True,
            'wandb_logging': False
        }
    }
    
    # Сохраняем конфигурацию
    config_path = os.path.join(project_dir, 'configs', 'training_config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ Конфигурация сохранена в {config_path}")
    
    return config

def create_dataset_yaml(project_dir: str):
    """Создание dataset.yaml для YOLOv8"""
    
    dataset_config = {
        'path': '/Users/bekzat/projects/AIinDrive/data/curated',
        'train': 'train',  # относительный путь от path
        'val': 'val',      # относительный путь от path
        'nc': 6,  # количество классов
        'names': {
            0: 'no_damage',
            1: 'rust', 
            2: 'dent',
            3: 'scratch',
            4: 'severe_damage',
            5: 'missing_part'
        }
    }
    
    dataset_path = os.path.join(project_dir, 'data', 'dataset.yaml')
    with open(dataset_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ Dataset config сохранён в {dataset_path}")
    return dataset_path

def main():
    """Основная функция подготовки"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    data_dir = os.path.join(project_dir, "data", "curated")
    annotation_file = os.path.join(project_dir, "data", "annotations", "coco", "instances.json")
    
    print("🚀 Подготовка данных для обучения модели AIinDrive")
    print("=" * 60)
    
    # 1. Валидация датасета
    validation_results = validate_dataset(data_dir, annotation_file)
    
    if validation_results['valid_images'] == 0:
        print("❌ Нет валидных изображений для обучения!")
        return
    
    # 2. Создание разбиения train/val
    train_files, val_files = create_train_val_split(data_dir, train_ratio=0.7)
    
    # 3. Создание YOLO аннотаций
    yolo_annotations_dir = os.path.join(project_dir, "data", "annotations", "yolo")
    create_yolo_annotations(annotation_file, yolo_annotations_dir, data_dir)
    
    # 4. Создание конфигурации обучения
    config = create_training_config(project_dir, train_files, val_files)
    
    # 5. Создание dataset.yaml
    dataset_yaml_path = create_dataset_yaml(project_dir)
    
    print("\n" + "=" * 60)
    print("✅ Подготовка данных завершена!")
    print("\nСледующие шаги:")
    print("1. Проверьте аннотации визуально")
    print("2. Запустите обучение: python scripts/train.py")
    print("3. Мониторьте процесс через TensorBoard")
    
    # Сохраняем результаты валидации
    results_path = os.path.join(project_dir, "data", "validation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Результаты валидации сохранены в {results_path}")

if __name__ == "__main__":
    main()
