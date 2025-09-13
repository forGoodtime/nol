#!/usr/bin/env python3
"""Простое обучение с использованием YOLOv8"""

import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import shutil

def setup_yolo_dataset(project_dir: str):
    """Подготовка датасета для YOLOv8"""
    
    print("📦 Подготовка датасета для YOLOv8...")
    
    # Создаём структуру для YOLOv8
    yolo_dir = os.path.join(project_dir, "data", "yolo_dataset")
    train_dir = os.path.join(yolo_dir, "images", "train")
    val_dir = os.path.join(yolo_dir, "images", "val")
    train_labels_dir = os.path.join(yolo_dir, "labels", "train")
    val_labels_dir = os.path.join(yolo_dir, "labels", "val")
    
    # Создаём директории
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Исходная директория с изображениями
    source_dir = os.path.join(project_dir, "data", "curated")
    yolo_annotations_dir = os.path.join(project_dir, "data", "annotations", "yolo")
    
    # Получаем список файлов
    image_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Разбиваем на train/val
    train_split = int(len(image_files) * 0.7)
    train_files = image_files[:train_split]
    val_files = image_files[train_split:]
    
    print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Копируем файлы
    for filename in train_files:
        # Копируем изображение
        src_img = os.path.join(source_dir, filename)
        dst_img = os.path.join(train_dir, filename)
        shutil.copy2(src_img, dst_img)
        
        # Копируем аннотацию
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        src_txt = os.path.join(yolo_annotations_dir, txt_filename)
        dst_txt = os.path.join(train_labels_dir, txt_filename)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            # Создаём пустой файл для изображений без аннотаций
            open(dst_txt, 'w').close()
    
    for filename in val_files:
        # Копируем изображение
        src_img = os.path.join(source_dir, filename)
        dst_img = os.path.join(val_dir, filename)
        shutil.copy2(src_img, dst_img)
        
        # Копируем аннотацию
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        src_txt = os.path.join(yolo_annotations_dir, txt_filename)
        dst_txt = os.path.join(val_labels_dir, txt_filename)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            # Создаём пустой файл для изображений без аннотаций
            open(dst_txt, 'w').close()
    
    # Создаём конфигурационный файл dataset.yaml для YOLOv8
    dataset_config = {
        'path': yolo_dir,
        'train': 'images/train',
        'val': 'images/val',
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
    
    dataset_yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Датасет подготовлен: {yolo_dir}")
    print(f"✅ Конфигурация: {dataset_yaml_path}")
    
    return dataset_yaml_path

def train_yolo_model(dataset_yaml_path: str, project_dir: str):
    """Обучение модели YOLOv8"""
    
    print("🚀 Начинаем обучение YOLOv8...")
    
    # Инициализация модели
    model = YOLO('yolov8n.pt')  # Используем nano модель для маленького датасета
    
    # Параметры обучения для маленького датасета
    training_args = {
        'data': dataset_yaml_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 2,  # Маленький batch для 7 изображений
        'patience': 20,
        'save': True,
        'cache': False,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 2,
        'project': os.path.join(project_dir, 'results', 'yolo_training'),
        'name': 'damage_detection_v1',
        'exist_ok': True,
        
        # Аугментации для маленького датасета
        'flipud': 0.2,
        'fliplr': 0.5,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.2,
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.1,
        'hsv_s': 0.3,
        'hsv_v': 0.3,
        'mixup': 0.1,
        'mosaic': 0.3,
        
        # Оптимизация для маленького датасета
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }
    
    print("📋 Параметры обучения:")
    print(f"  Модель: YOLOv8n")
    print(f"  Эпохи: {training_args['epochs']}")
    print(f"  Batch size: {training_args['batch']}")
    print(f"  Устройство: {training_args['device']}")
    print(f"  Размер изображения: {training_args['imgsz']}")
    
    # Запуск обучения
    try:
        results = model.train(**training_args)
        print("🎉 Обучение завершено успешно!")
        
        # Путь к лучшей модели
        best_model_path = os.path.join(
            project_dir, 'results', 'yolo_training', 
            'damage_detection_v1', 'weights', 'best.pt'
        )
        
        print(f"🏆 Лучшая модель: {best_model_path}")
        
        # Валидация модели
        if os.path.exists(best_model_path):
            print("🔍 Запуск валидации...")
            best_model = YOLO(best_model_path)
            val_results = best_model.val(data=dataset_yaml_path)
            print("✅ Валидация завершена")
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ Ошибка во время обучения: {str(e)}")
        raise

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    
    print("🚀 Обучение модели детекции повреждений AIinDrive")
    print("=" * 60)
    
    # Проверяем наличие данных
    if not os.path.exists(os.path.join(project_dir, "data", "curated")):
        print("❌ Данные не найдены!")
        print("💡 Запустите: python scripts/prepare_training.py")
        return
    
    # Проверяем CUDA
    if torch.cuda.is_available():
        print(f"🔥 CUDA доступна: {torch.cuda.get_device_name()}")
    else:
        print("💻 Используется CPU")
    
    try:
        # 1. Подготовка датасета
        dataset_yaml_path = setup_yolo_dataset(project_dir)
        
        # 2. Обучение модели
        best_model_path = train_yolo_model(dataset_yaml_path, project_dir)
        
        print("\n" + "=" * 60)
        print("🎉 Обучение завершено успешно!")
        print(f"🏆 Лучшая модель: {best_model_path}")
        print("\n📊 Проверьте результаты в папке results/yolo_training/")
        print("💡 Для тестирования модели запустите: python scripts/test_trained_model.py")
        
    except Exception as e:
        print(f"💥 Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
