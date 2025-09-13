#!/usr/bin/env python3
"""
Скрипт для подготовки данных
"""
import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import random
import logging

# Добавляем корневую директорию в path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def create_train_val_test_splits(
    images_dir: str,
    annotations_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Создает разделение данных на train/val/test
    
    Args:
        images_dir: папка с изображениями
        annotations_file: файл аннотаций COCO
        output_dir: папка для сохранения списков файлов
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
        test_ratio: доля тестовой выборки
        seed: random seed для воспроизводимости
        
    Returns:
        словарь со списками файлов для каждого набора
    """
    random.seed(seed)
    
    # Загружаем аннотации COCO
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Получаем список всех изображений
    all_images = []
    for img_info in coco_data['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        if os.path.exists(img_path):
            all_images.append(img_info['file_name'])
    
    # Перемешиваем
    random.shuffle(all_images)
    
    # Вычисляем размеры наборов
    total_images = len(all_images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size
    
    # Разделяем
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    # Создаем выходную папку
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем списки
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, image_list in splits.items():
        split_file = output_path / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for img_name in image_list:
                f.write(f"{img_name}\n")
    
    logger.info(f"Создано разделение данных:")
    logger.info(f"  Train: {len(train_images)} изображений")
    logger.info(f"  Val: {len(val_images)} изображений")
    logger.info(f"  Test: {len(test_images)} изображений")
    
    return splits


def validate_dataset(images_dir: str, annotations_file: str) -> Dict[str, Any]:
    """
    Валидация датасета - проверка корректности аннотаций и изображений
    
    Args:
        images_dir: папка с изображениями
        annotations_file: файл аннотаций COCO
        
    Returns:
        отчет о валидации
    """
    logger.info("Начинаем валидацию датасета...")
    
    # Загружаем аннотации
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    report = {
        'total_images': 0,
        'valid_images': 0,
        'missing_images': [],
        'corrupted_images': [],
        'annotations_count': 0,
        'categories': {},
        'damage_levels': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    # Проверяем изображения
    for img_info in coco_data['images']:
        report['total_images'] += 1
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            report['missing_images'].append(img_info['file_name'])
            continue
        
        # Проверяем, что файл можно открыть
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img.verify()
            report['valid_images'] += 1
        except Exception as e:
            report['corrupted_images'].append({
                'file': img_info['file_name'],
                'error': str(e)
            })
    
    # Анализируем аннотации
    for annotation in coco_data.get('annotations', []):
        report['annotations_count'] += 1
        
        category_id = annotation['category_id']
        if category_id not in report['categories']:
            report['categories'][category_id] = 0
        report['categories'][category_id] += 1
    
    # Анализируем категории по уровням повреждений
    for category in coco_data.get('categories', []):
        damage_level = category.get('damage_level', 0)
        if damage_level in report['damage_levels']:
            count = report['categories'].get(category['id'], 0)
            report['damage_levels'][damage_level] += count
    
    logger.info("Валидация завершена:")
    logger.info(f"  Всего изображений: {report['total_images']}")
    logger.info(f"  Валидных изображений: {report['valid_images']}")
    logger.info(f"  Отсутствующих: {len(report['missing_images'])}")
    logger.info(f"  Поврежденных: {len(report['corrupted_images'])}")
    logger.info(f"  Аннотаций: {report['annotations_count']}")
    
    return report


def create_sample_coco_annotation(
    images_dir: str,
    output_file: str,
    num_samples: int = 10
) -> None:
    """
    Создает пример файла аннотаций COCO для тестирования
    
    Args:
        images_dir: папка с изображениями
        output_file: путь к выходному файлу аннотаций
        num_samples: количество примеров для создания
    """
    from PIL import Image
    import datetime
    
    # Базовая структура COCO
    coco_data = {
        "info": {
            "description": "AIinDrive Vehicle Damage Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "AIinDrive Team",
            "date_created": datetime.datetime.now().isoformat()
        },
        "categories": [
            {"id": 1, "name": "rust", "damage_level": 2},
            {"id": 2, "name": "dent", "damage_level": 2},
            {"id": 3, "name": "scratch", "damage_level": 1},
            {"id": 4, "name": "missing_part", "damage_level": 3},
            {"id": 5, "name": "corrosion", "damage_level": 2},
            {"id": 6, "name": "dirt", "damage_level": 1}
        ],
        "images": [],
        "annotations": []
    }
    
    # Находим изображения в папке
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    images_path = Path(images_dir)
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))
    
    # Берем первые num_samples изображений
    selected_images = image_files[:num_samples]
    
    annotation_id = 1
    
    for img_id, img_path in enumerate(selected_images, 1):
        try:
            # Получаем информацию об изображении
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Добавляем информацию об изображении
            coco_data['images'].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Создаем примерные аннотации (случайные bbox)
            num_annotations = random.randint(0, 3)  # 0-3 повреждения на изображение
            
            for _ in range(num_annotations):
                # Случайный bbox
                x = random.randint(0, width // 2)
                y = random.randint(0, height // 2)
                w = random.randint(50, width // 4)
                h = random.randint(50, height // 4)
                
                # Убеждаемся, что bbox в пределах изображения
                x = min(x, width - w)
                y = min(y, height - h)
                
                # Случайная категория
                category_id = random.randint(1, 6)
                
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                
        except Exception as e:
            logger.warning(f"Пропускаем {img_path}: {e}")
    
    # Сохраняем аннотации
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"Создан пример файла аннотаций: {output_file}")
    logger.info(f"  Изображений: {len(coco_data['images'])}")
    logger.info(f"  Аннотаций: {len(coco_data['annotations'])}")


def copy_and_organize_data(source_dir: str, target_dir: str) -> None:
    """
    Копирует и организует данные в структуру проекта
    
    Args:
        source_dir: исходная папка с данными
        target_dir: целевая папка проекта
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Создаем структуру папок
    curated_dir = target_path / "data" / "curated"
    annotations_dir = target_path / "data" / "annotations" / "coco"
    
    curated_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем изображения
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    copied_count = 0
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            target_file = curated_dir / file_path.name
            shutil.copy2(file_path, target_file)
            copied_count += 1
    
    logger.info(f"Скопировано {copied_count} изображений в {curated_dir}")
    
    # Создаем пример аннотаций
    create_sample_coco_annotation(
        str(curated_dir),
        str(annotations_dir / "instances.json")
    )


def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(description="Подготовка данных для обучения")
    parser.add_argument("--action", required=True,
                       choices=['split', 'validate', 'organize', 'create_sample'],
                       help="Действие для выполнения")
    parser.add_argument("--images-dir", 
                       help="Папка с изображениями")
    parser.add_argument("--annotations",
                       help="Файл аннотаций COCO")
    parser.add_argument("--output-dir", default="data/splits",
                       help="Папка для сохранения результатов")
    parser.add_argument("--source-dir",
                       help="Исходная папка с данными (для organize)")
    parser.add_argument("--target-dir", default=".",
                       help="Целевая папка проекта (для organize)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Доля обучающей выборки")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Доля валидационной выборки")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Доля тестовой выборки")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настройка логирования
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        if args.action == 'split':
            if not args.images_dir or not args.annotations:
                parser.error("Для split требуются --images-dir и --annotations")
            
            create_train_val_test_splits(
                args.images_dir,
                args.annotations,
                args.output_dir,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.seed
            )
        
        elif args.action == 'validate':
            if not args.images_dir or not args.annotations:
                parser.error("Для validate требуются --images-dir и --annotations")
            
            report = validate_dataset(args.images_dir, args.annotations)
            
            # Сохраняем отчет
            report_file = Path(args.output_dir) / "validation_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Отчет о валидации сохранен: {report_file}")
        
        elif args.action == 'organize':
            if not args.source_dir:
                parser.error("Для organize требуется --source-dir")
            
            copy_and_organize_data(args.source_dir, args.target_dir)
        
        elif args.action == 'create_sample':
            if not args.images_dir:
                parser.error("Для create_sample требуется --images-dir")
            
            output_file = Path(args.output_dir) / "sample_annotations.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            create_sample_coco_annotation(args.images_dir, str(output_file))
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
