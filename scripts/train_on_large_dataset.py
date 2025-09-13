#!/usr/bin/env python3
"""Обучение на большом датасете повреждений автомобилей"""

import os
import sys
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def train_on_large_dataset(project_dir: str, dataset_yaml_path: str, config_path: str):
    """Обучение модели на большом датасете"""
    
    print("🚀 Обучение на большом датасете повреждений автомобилей")
    print("=" * 70)
    
    # Загружаем конфигурацию
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Проверяем датасет
    if not os.path.exists(dataset_yaml_path):
        print(f"❌ Датасет не найден: {dataset_yaml_path}")
        return None
    
    print(f"📊 Датасет: {dataset_yaml_path}")
    
    # Считаем количество изображений
    with open(dataset_yaml_path, 'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_path = dataset_config['path']
    train_images_path = os.path.join(dataset_path, 'images')
    
    if os.path.exists(train_images_path):
        image_files = [f for f in os.listdir(train_images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(image_files)
        print(f"📈 Найдено изображений: {num_images}")
    else:
        num_images = 0
        print("⚠️ Не удалось подсчитать количество изображений")
    
    if num_images < 100:
        print("⚠️ Маленький датасет - используем специальные настройки")
        # Корректируем параметры для маленького датасета
        config['training_config']['batch_size'] = min(8, max(1, num_images // 10))
        config['training_config']['epochs'] = 150
    
    # Выбираем модель в зависимости от размера датасета
    if num_images < 500:
        model_name = 'yolov8n.pt'  # Nano для маленьких датасетов
        print("🏗️ Используем YOLOv8n (nano) для маленького датасета")
    elif num_images < 2000:
        model_name = 'yolov8s.pt'  # Small для средних датасетов
        print("🏗️ Используем YOLOv8s (small) для среднего датасета")
    else:
        model_name = 'yolov8m.pt'  # Medium для больших датасетов
        print("🏗️ Используем YOLOv8m (medium) для большого датасета")
    
    # Инициализация модели
    model = YOLO(model_name)
    
    # Настройка устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Устройство: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Памяти: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Параметры обучения
    training_args = {
        'data': dataset_yaml_path,
        'epochs': config['training_config']['epochs'],
        'imgsz': config['model_config']['input_size'],
        'batch': config['training_config']['batch_size'],
        'patience': config['training_config']['patience'],
        'save': True,
        'cache': False,
        'device': device,
        'workers': config['data_config']['num_workers'],
        'project': os.path.join(project_dir, 'results', 'large_dataset_training'),
        'name': f'car_damage_v2_{num_images}imgs',
        'exist_ok': True,
        'verbose': True,
        
        # Аугментации из конфига
        'flipud': config['augmentation_config'].get('vertical_flip', 0.1),
        'fliplr': config['augmentation_config'].get('horizontal_flip', 0.5),
        'degrees': config['augmentation_config'].get('rotation', 10),
        'translate': config['augmentation_config'].get('translate', 0.1),
        'scale': config['augmentation_config'].get('scale', 0.5),
        'shear': config['augmentation_config'].get('shear', 2.0),
        'perspective': config['augmentation_config'].get('perspective', 0.0),
        'hsv_h': config['augmentation_config'].get('hue', 0.015),
        'hsv_s': config['augmentation_config'].get('saturation', 0.7),
        'hsv_v': config['augmentation_config'].get('brightness', 0.4),
        'mixup': config['augmentation_config'].get('mixup', 0.15),
        'mosaic': config['augmentation_config'].get('mosaic', 1.0),
        'copy_paste': config['augmentation_config'].get('copy_paste', 0.3),
        
        # Оптимизация
        'lr0': config['training_config']['learning_rate'],
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': config['training_config']['weight_decay'],
        'warmup_epochs': config['training_config'].get('warmup_epochs', 3),
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': config['training_config'].get('cos_lr', False),
        
        # Валидация
        'val': True,
        'conf': config['validation_config']['conf_threshold'],
        'iou': config['validation_config']['iou_threshold'],
        'max_det': config['validation_config'].get('max_det', 300),
        
        # Логирование
        'plots': True,
        'save_period': config['logging_config'].get('save_period', -1)
    }
    
    print("\n📋 Параметры обучения:")
    print(f"  Модель: {model_name}")
    print(f"  Изображений в датасете: {num_images}")
    print(f"  Эпох: {training_args['epochs']}")
    print(f"  Batch size: {training_args['batch']}")
    print(f"  Learning rate: {training_args['lr0']}")
    print(f"  Размер изображения: {training_args['imgsz']}")
    print(f"  Workers: {training_args['workers']}")
    print(f"  Patience: {training_args['patience']}")
    
    # Начинаем обучение
    try:
        print(f"\n🎯 Начинаем обучение...")
        print(f"⏰ Время начала: {datetime.now().strftime('%H:%M:%S')}")
        
        results = model.train(**training_args)
        
        print(f"\n🎉 Обучение завершено!")
        print(f"⏰ Время окончания: {datetime.now().strftime('%H:%M:%S')}")
        
        # Путь к лучшей модели
        best_model_path = os.path.join(
            project_dir, 'results', 'large_dataset_training',
            f'car_damage_v2_{num_images}imgs', 'weights', 'best.pt'
        )
        
        print(f"🏆 Лучшая модель: {best_model_path}")
        
        # Валидация модели
        if os.path.exists(best_model_path):
            print("\n🔍 Запуск финальной валидации...")
            best_model = YOLO(best_model_path)
            val_results = best_model.val(data=dataset_yaml_path)
            
            print("✅ Валидация завершена")
            print(f"📊 Результаты валидации: {val_results}")
        
        # Сохраняем информацию о тренировке
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_yaml_path,
            'dataset_size': num_images,
            'model_name': model_name,
            'config_used': config,
            'training_args': training_args,
            'best_model_path': best_model_path,
            'device': device,
            'status': 'completed'
        }
        
        info_path = os.path.join(
            project_dir, 'results', 'large_dataset_training',
            f'training_info_{num_images}imgs.json'
        )
        
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Информация о тренировке: {info_path}")
        
        return best_model_path
        
    except Exception as e:
        print(f"\n💥 Ошибка во время обучения: {str(e)}")
        
        # Сохраняем информацию об ошибке
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_yaml_path,
            'dataset_size': num_images,
            'model_name': model_name,
            'config_used': config,
            'error': str(e),
            'status': 'failed'
        }
        
        error_path = os.path.join(
            project_dir, 'results', 'large_dataset_training',
            f'training_error_{num_images}imgs.json'
        )
        
        os.makedirs(os.path.dirname(error_path), exist_ok=True)
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        print(f"❌ Информация об ошибке: {error_path}")
        raise

def main():
    """Основная функция"""
    
    project_dir = "/Users/bekzat/projects/AIinDrive"
    
    # Проверяем наличие синтетического датасета
    synthetic_dataset_yaml = os.path.join(
        project_dir, "data", "synthetic_car_damage", "dataset.yaml"
    )
    
    large_config_path = os.path.join(
        project_dir, "configs", "large_dataset_training_config.yaml"
    )
    
    if not os.path.exists(synthetic_dataset_yaml):
        print("❌ Синтетический датасет не найден!")
        print("💡 Запустите сначала: python scripts/setup_large_datasets.py")
        return
    
    if not os.path.exists(large_config_path):
        print("❌ Конфигурация не найдена!")
        print("💡 Запустите сначала: python scripts/setup_large_datasets.py")
        return
    
    # Запускаем обучение
    best_model_path = train_on_large_dataset(
        project_dir, 
        synthetic_dataset_yaml, 
        large_config_path
    )
    
    if best_model_path:
        print(f"\n🏆 Обучение успешно завершено!")
        print(f"📁 Лучшая модель: {best_model_path}")
        print(f"\n💡 Следующие шаги:")
        print(f"1. Проверьте результаты в results/large_dataset_training/")
        print(f"2. Протестируйте модель: python scripts/test_large_model.py")
        print(f"3. Интегрируйте в CV анализ для сравнения с традиционными методами")

if __name__ == "__main__":
    main()
