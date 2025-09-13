"""
Модуль для инференса - обработка одного изображения или батча изображений
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

# Добавляем корневую директорию в path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from src.models.backbone import get_backbone
from src.models.classifier import DamageClassifier
from src.models.segmenter import get_segmentation_model
from src.models.anomaly_module import get_anomaly_detector
from src.utils.shadow_removal import ShadowRemover
from src.datasets.augmentation import get_test_transforms
from src.utils.visualize import overlay_heatmap, create_damage_visualization

logger = logging.getLogger(__name__)


class DamageDetectionInference:
    """Класс для инференса системы обнаружения повреждений"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any],
        device: str = "cuda",
        use_shadow_removal: bool = True
    ):
        """
        Args:
            checkpoint_path: путь к файлу весов модели
            config: конфигурация модели
            device: устройство для вычислений
            use_shadow_removal: использовать ли предобработку теней
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_shadow_removal = use_shadow_removal
        
        # Инициализируем компоненты
        self.shadow_remover = ShadowRemover() if use_shadow_removal else None
        self.transforms = get_test_transforms(tuple(config['data']['image_size']))
        
        # Загружаем модели
        self._load_models(checkpoint_path)
        
        # Классы повреждений
        self.damage_classes = {
            0: "No damage",
            1: "Light damage",
            2: "Moderate damage", 
            3: "Severe damage"
        }
        
        logger.info(f"Модель загружена на устройство: {self.device}")
    
    def _load_models(self, checkpoint_path: str):
        """Загружает обученные модели"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Классификатор основной модели
            self.classifier = DamageClassifier(
                backbone_name=self.config['model']['backbone'],
                num_classes=self.config['model']['num_classes'],
                pretrained=False
            ).to(self.device)
            
            # Загружаем веса классификатора
            if 'classifier_state_dict' in checkpoint:
                self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            else:
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
            
            self.classifier.eval()
            
            # Сегментация (если включена)
            if self.config['model']['segmentation']['enabled']:
                self.segmenter = get_segmentation_model(
                    architecture=self.config['model']['segmentation']['architecture'],
                    backbone=self.config['model']['backbone']
                ).to(self.device)
                
                if 'segmenter_state_dict' in checkpoint:
                    self.segmenter.load_state_dict(checkpoint['segmenter_state_dict'])
                    
                self.segmenter.eval()
            else:
                self.segmenter = None
            
            # Anomaly detection (если включен)
            if self.config['model']['anomaly']['enabled']:
                self.anomaly_detector = get_anomaly_detector(
                    method=self.config['model']['anomaly']['method']
                ).to(self.device)
                
                if 'anomaly_state_dict' in checkpoint:
                    self.anomaly_detector.load_state_dict(checkpoint['anomaly_state_dict'])
                    
                self.anomaly_detector.eval()
            else:
                self.anomaly_detector = None
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Предобработка изображения перед инференсом
        
        Args:
            image: PIL изображение
            
        Returns:
            тензор для подачи в модель
        """
        # Конвертируем в numpy
        img_array = np.array(image)
        
        # Предобработка теней и бликов
        if self.shadow_remover:
            img_array = self.shadow_remover.process_image_pipeline(img_array)
        
        # Конвертируем обратно в PIL для трансформаций
        processed_image = Image.fromarray(img_array)
        
        # Применяем трансформации
        tensor = self.transforms(image=np.array(processed_image))['image']
        
        return tensor.unsqueeze(0)  # добавляем batch dimension
    
    def predict_classification(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Классификация степени повреждения
        
        Args:
            image_tensor: предобработанное изображение
            
        Returns:
            результаты классификации
        """
        with torch.no_grad():
            logits = self.classifier(image_tensor.to(self.device))
            probabilities = F.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Все вероятности по классам
            class_probs = {
                self.damage_classes[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
            return {
                'predicted_class': predicted_class,
                'class_name': self.damage_classes[predicted_class],
                'confidence': confidence,
                'class_probabilities': class_probs
            }
    
    def predict_segmentation(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Сегментация повреждений
        
        Args:
            image_tensor: предобработанное изображение
            
        Returns:
            маска сегментации или None
        """
        if not self.segmenter:
            return None
        
        with torch.no_grad():
            mask_logits = self.segmenter(image_tensor.to(self.device))
            mask_probs = torch.sigmoid(mask_logits)
            
            # Конвертируем в numpy маску
            mask = mask_probs[0, 0].cpu().numpy()  # берем первый канал первого изображения
            
            return mask
    
    def predict_anomaly(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Обнаружение аномалий
        
        Args:
            image_tensor: предобработанное изображение
            
        Returns:
            heatmap аномалий или None
        """
        if not self.anomaly_detector:
            return None
        
        with torch.no_grad():
            anomaly_score, anomaly_map = self.anomaly_detector(image_tensor.to(self.device))
            
            if anomaly_map is not None:
                return anomaly_map[0].cpu().numpy()  # первое изображение в батче
            
            return None
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Полный инференс на изображении
        
        Args:
            image_path: путь к изображению
            
        Returns:
            полные результаты анализа
        """
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            logger.info(f"Обработка изображения: {image_path}")
            
            # Предобработка
            image_tensor = self.preprocess_image(image)
            
            # Классификация
            classification_results = self.predict_classification(image_tensor)
            
            # Сегментация
            segmentation_mask = self.predict_segmentation(image_tensor)
            
            # Anomaly detection
            anomaly_heatmap = self.predict_anomaly(image_tensor)
            
            # Объединяем результаты
            results = {
                'image_path': image_path,
                'original_size': original_size,
                'classification': classification_results,
                'segmentation': {
                    'has_mask': segmentation_mask is not None,
                    'mask_shape': segmentation_mask.shape if segmentation_mask is not None else None
                },
                'anomaly': {
                    'has_heatmap': anomaly_heatmap is not None,
                    'heatmap_shape': anomaly_heatmap.shape if anomaly_heatmap is not None else None
                }
            }
            
            # Создаем визуализации
            visualizations = self._create_visualizations(
                image, segmentation_mask, anomaly_heatmap, classification_results
            )
            
            results['visualizations'] = visualizations
            
            logger.info(f"Предсказание: {classification_results['class_name']} "
                       f"(уверенность: {classification_results['confidence']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {image_path}: {e}")
            raise
    
    def _create_visualizations(
        self, 
        original_image: Image.Image,
        segmentation_mask: Optional[np.ndarray],
        anomaly_heatmap: Optional[np.ndarray],
        classification_results: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Создает визуализации результатов"""
        
        visualizations = {}
        img_array = np.array(original_image)
        
        # Основная визуализация с наложением
        if segmentation_mask is not None or anomaly_heatmap is not None:
            # Используем маску сегментации если есть, иначе heatmap аномалий
            mask_to_use = segmentation_mask if segmentation_mask is not None else anomaly_heatmap
            
            if mask_to_use is not None:
                # Изменяем размер маски до размера оригинального изображения
                mask_resized = cv2.resize(mask_to_use, original_image.size)
                
                # Создаем наложение
                overlay = overlay_heatmap(img_array, mask_resized)
                visualizations['overlay'] = overlay
                
                # Сохраняем отдельно маску/heatmap
                heatmap_colored = cv2.applyColorMap(
                    (mask_resized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                visualizations['heatmap'] = heatmap_colored
        
        # Визуализация с информацией о классификации
        damage_viz = create_damage_visualization(
            img_array, 
            classification_results,
            mask_to_use if 'mask_to_use' in locals() else None
        )
        visualizations['damage_info'] = damage_viz
        
        return visualizations
    
    def batch_predict(self, image_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """
        Батчевая обработка изображений
        
        Args:
            image_paths: список путей к изображениям
            output_dir: директория для сохранения результатов
            
        Returns:
            список результатов для каждого изображения
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Обработка {i+1}/{len(image_paths)}: {image_path}")
                
                result = self.predict(image_path)
                results.append(result)
                
                # Сохраняем результаты
                image_name = Path(image_path).stem
                
                # JSON с результатами
                json_path = output_path / f"{image_name}_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    # Убираем numpy массивы из JSON
                    json_result = {k: v for k, v in result.items() if k != 'visualizations'}
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
                
                # Сохраняем визуализации
                if 'visualizations' in result:
                    for viz_name, viz_image in result['visualizations'].items():
                        viz_path = output_path / f"{image_name}_{viz_name}.png"
                        cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                logger.error(f"Ошибка обработки {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def main():
    """CLI интерфейс для инференса"""
    parser = argparse.ArgumentParser(description="Инференс модели обнаружения повреждений")
    parser.add_argument("--input", "-i", required=True, 
                       help="Путь к изображению или папке с изображениями")
    parser.add_argument("--model", "-m", required=True,
                       help="Путь к файлу модели (.pt)")
    parser.add_argument("--config", "-c", default="src/config/default.yaml",
                       help="Путь к файлу конфигурации")
    parser.add_argument("--output", "-o", default="outputs/inference",
                       help="Папка для сохранения результатов")
    parser.add_argument("--device", default="cuda",
                       help="Устройство: cuda или cpu")
    parser.add_argument("--no-shadow-removal", action="store_true",
                       help="Отключить предобработку теней")
    parser.add_argument("--batch", action="store_true",
                       help="Батчевая обработка всех изображений в папке")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настройка логирования
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Загружаем конфигурацию
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Создаем инферер
        inferencer = DamageDetectionInference(
            checkpoint_path=args.model,
            config=config,
            device=args.device,
            use_shadow_removal=not args.no_shadow_removal
        )
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Обработка одного файла
            result = inferencer.predict(str(input_path))
            
            # Сохраняем результат
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = input_path.stem
            json_path = output_dir / f"{image_name}_results.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json_result = {k: v for k, v in result.items() if k != 'visualizations'}
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            # Сохраняем visualizations
            if 'visualizations' in result:
                for viz_name, viz_image in result['visualizations'].items():
                    viz_path = output_dir / f"{image_name}_{viz_name}.png"
                    cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
            
            print(f"Результаты сохранены в {output_dir}")
            print(f"Предсказание: {result['classification']['class_name']} "
                  f"(уверенность: {result['classification']['confidence']:.3f})")
        
        elif input_path.is_dir() or args.batch:
            # Батчевая обработка
            if input_path.is_dir():
                # Находим все изображения в папке
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                image_paths = []
                
                for ext in image_extensions:
                    image_paths.extend(input_path.glob(f"*{ext}"))
                    image_paths.extend(input_path.glob(f"*{ext.upper()}"))
                
                image_paths = [str(p) for p in image_paths]
            else:
                image_paths = [str(input_path)]
            
            if not image_paths:
                logger.error("Изображения не найдены")
                return
            
            logger.info(f"Найдено {len(image_paths)} изображений для обработки")
            
            # Обрабатываем батч
            results = inferencer.batch_predict(image_paths, args.output)
            
            # Статистика
            successful = sum(1 for r in results if 'error' not in r)
            failed = len(results) - successful
            
            logger.info(f"Обработка завершена: {successful} успешно, {failed} с ошибками")
            
        else:
            logger.error(f"Путь не найден: {input_path}")
            return
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
