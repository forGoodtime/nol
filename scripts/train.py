#!/usr/bin/env python3
"""Обучение модели детекции повреждений"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
import json

# Добавляем корневую директорию в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainers.damage_trainer import DamageTrainer
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

def check_small_dataset_setup(config):
    """Проверка и настройка для маленького датасета"""
    
    # Считаем количество изображений
    data_dir = config.get('data_config', {}).get('dataset_path', 'data/curated')
    if os.path.exists(data_dir):
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(image_files)
        
        print(f"📊 Найдено {num_images} изображений в датасете")
        
        # Настройки для маленького датасета
        if num_images < 50:
            print("⚠️  Обнаружен маленький датасет. Применяем специальные настройки:")
            
            # Уменьшаем batch size
            if config['training_config']['batch_size'] > num_images // 2:
                config['training_config']['batch_size'] = max(1, num_images // 3)
                print(f"  🔧 Batch size: {config['training_config']['batch_size']}")
            
            # Увеличиваем аугментации
            config['augmentation_config']['horizontal_flip'] = 0.8
            config['augmentation_config']['rotation'] = 15
            config['augmentation_config']['brightness'] = 0.3
            config['augmentation_config']['contrast'] = 0.3
            print("  🔧 Усилены аугментации данных")
            
            # Настройки для transfer learning
            config['model_config']['freeze_backbone'] = True
            config['model_config']['freeze_epochs'] = 20
            print("  🔧 Заморозка backbone на 20 эпох")
            
            # Увеличиваем количество эпох
            config['training_config']['epochs'] = 200
            print("  🔧 Увеличено количество эпох: 200")
            
            # Усиливаем regularization
            config['training_config']['weight_decay'] = 0.001
            config['training_config']['dropout'] = 0.3
            print("  🔧 Усилена регуляризация")
    
    return config

def create_minimal_dataset_structure(project_dir: str):
    """Создание минимальной структуры для маленького датасета"""
    
    data_dir = os.path.join(project_dir, "data", "curated")
    
    # Создаём train и val директории для YOLO
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Копируем изображения
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 70% в train, 30% в val
    n_train = max(1, int(len(image_files) * 0.7))
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"📦 Создание структуры: {len(train_files)} train, {len(val_files)} val")
    
    # Копируем файлы
    import shutil
    for f in train_files:
        src = os.path.join(data_dir, f)
        dst = os.path.join(train_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    for f in val_files:
        src = os.path.join(data_dir, f)
        dst = os.path.join(val_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    return len(train_files), len(val_files)

def main():
    """Основная функция обучения"""
    
    # Настройка логирования
    logger = setup_logger("training", level=logging.INFO)
    logger.info("🚀 Запуск обучения модели детекции повреждений AIinDrive")
    
    try:
        # Загрузка конфигурации
        config_path = "configs/training_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"❌ Файл конфигурации не найден: {config_path}")
            logger.info("💡 Запустите сначала: python scripts/prepare_training.py")
            return
        
        config = load_config(config_path)
        logger.info(f"✅ Конфигурация загружена из {config_path}")
        
        # Проверка датасета и адаптация для маленького размера
        config = check_small_dataset_setup(config)
        
        # Создание структуры датасета
        project_dir = str(Path(__file__).parent.parent)
        n_train, n_val = create_minimal_dataset_structure(project_dir)
        
        if n_train == 0:
            logger.error("❌ Нет данных для обучения!")
            return
        
        # Проверка CUDA
        if torch.cuda.is_available():
            logger.info(f"🔥 CUDA доступна: {torch.cuda.get_device_name()}")
            device = torch.device("cuda")
        else:
            logger.info("💻 Используется CPU")
            device = torch.device("cpu")
        
        # Инициализация тренера
        logger.info("🔄 Инициализация тренера...")
        trainer = DamageTrainer(config, device=device)
        
        # Информация о тренировке
        logger.info("📋 Параметры обучения:")
        logger.info(f"  Модель: {config['model_config']['architecture']}")
        logger.info(f"  Классов: {config['model_config']['num_classes']}")
        logger.info(f"  Batch size: {config['training_config']['batch_size']}")
        logger.info(f"  Эпох: {config['training_config']['epochs']}")
        logger.info(f"  Learning rate: {config['training_config']['learning_rate']}")
        logger.info(f"  Train images: {n_train}")
        logger.info(f"  Val images: {n_val}")
        
        # Запуск обучения
        logger.info("🎯 Начинаем обучение...")
        best_model_path = trainer.train()
        
        logger.info("🎉 Обучение завершено!")
        logger.info(f"🏆 Лучшая модель: {best_model_path}")
        
        # Сохраняем информацию о тренировке
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'dataset_size': {'train': n_train, 'val': n_val},
            'best_model_path': best_model_path,
            'device': str(device)
        }
        
        info_path = os.path.join(project_dir, 'results', 'last_training_info.json')
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Информация о тренировке сохранена: {info_path}")
        
    except Exception as e:
        logger.error(f"💥 Ошибка во время обучения: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
