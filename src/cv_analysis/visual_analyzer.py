"""
Визуализатор результатов анализа повреждений
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from pathlib import Path
import logging

from .damage_detector import DamageRegion

logger = logging.getLogger(__name__)


class VisualAnalyzer:
    """
    Класс для визуализации результатов CV анализа повреждений
    """
    
    def __init__(self):
        # Цветовая схема для разных типов повреждений
        self.damage_colors = {
            'rust_type_1': (0, 100, 255),      # Оранжево-красный для ржавчины
            'rust_type_2': (0, 140, 255),      # Красный для темной ржавчины
            'rust_type_3': (0, 180, 255),      # Ярко-красный для светлой ржавчины
            'dent': (0, 0, 255),               # Красный для вмятин
            'scratch': (0, 165, 255),          # Оранжевый для царапин
            'missing_part': (0, 0, 139),       # Темно-красный для недостающих частей
        }
        
        # Цвета для уровней серьезности
        self.severity_colors = {
            1: (0, 255, 0),      # Зеленый для легких повреждений
            2: (0, 165, 255),    # Оранжевый для средних повреждений
            3: (0, 0, 255),      # Красный для серьезных повреждений
        }
        
        # Настройки визуализации
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def create_damage_mask(self, image: np.ndarray, regions: List[DamageRegion]) -> np.ndarray:
        """
        Создание цветовой маски повреждений
        
        Args:
            image: исходное изображение
            regions: список областей повреждений
            
        Returns:
            Цветовая маска с выделенными областями повреждений
        """
        # Создаем маску того же размера что и изображение
        mask = np.zeros_like(image)
        
        for region in regions:
            x, y, w, h = region.bbox
            color = self.severity_colors.get(region.severity, (128, 128, 128))
            
            # Если есть контур, используем его для более точной маски
            if region.contour is not None:
                cv2.fillPoly(mask, [region.contour], color)
            else:
                # Иначе используем прямоугольник
                cv2.rectangle(mask, (x, y), (x + w, y + h), color, -1)
        
        return mask
    
    def create_analysis_overlay(self, image: np.ndarray, cv_results: Dict) -> np.ndarray:
        """
        Создание изображения с наложением результатов CV анализа
        
        Args:
            image: исходное изображение
            cv_results: результаты CV анализа
            
        Returns:
            Изображение с наложенными результатами анализа
        """
        overlay = image.copy()
        regions = cv_results['damage_regions']
        
        # Создаем полупрозрачную маску
        mask = self.create_damage_mask(image, regions)
        
        # Накладываем маску с прозрачностью
        alpha = 0.3
        overlay = cv2.addWeighted(overlay, 1 - alpha, mask, alpha, 0)
        
        # Добавляем bounding boxes и подписи
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            
            # Цвет рамки по типу повреждения
            color = self.damage_colors.get(region.damage_type, (255, 255, 255))
            
            # Рисуем рамку
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, self.thickness)
            
            # Подготавливаем текст
            damage_type = region.damage_type.replace('_', ' ').title()
            confidence_text = f'{region.confidence:.2f}'
            severity_text = f'L{region.severity}'
            
            # Основная подпись
            label = f'{damage_type} ({confidence_text})'
            
            # Вычисляем размер текста для фона
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            
            # Рисуем фон для текста
            cv2.rectangle(overlay, 
                         (x, y - text_height - baseline - 5),
                         (x + text_width + 10, y),
                         color, -1)
            
            # Рисуем текст
            cv2.putText(overlay, label, (x + 5, y - baseline - 2),
                       self.font, self.font_scale, (255, 255, 255), 1)
            
            # Добавляем индикатор серьезности
            severity_color = self.severity_colors[region.severity]
            cv2.circle(overlay, (x + w - 15, y + 15), 10, severity_color, -1)
            cv2.putText(overlay, str(region.severity), (x + w - 20, y + 20),
                       self.font, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def create_summary_visualization(self, original: np.ndarray, cv_results: Dict, 
                                   ml_results: Optional[Dict] = None) -> Figure:
        """
        Создание сводной визуализации с оригиналом и анализом
        
        Args:
            original: оригинальное изображение
            cv_results: результаты CV анализа
            ml_results: результаты ML анализа (опционально)
            
        Returns:
            Figure с визуализацией
        """
        # Определяем количество колонок
        n_cols = 3 if ml_results else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 8))
        
        if n_cols == 2:
            axes = [axes[0], axes[1]]
        
        # Конвертируем BGR в RGB для matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cv_overlay = self.create_analysis_overlay(original, cv_results)
        cv_overlay_rgb = cv2.cvtColor(cv_overlay, cv2.COLOR_BGR2RGB)
        
        # 1. Оригинальное изображение
        axes[0].imshow(original_rgb)
        axes[0].set_title('Оригинал', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. CV анализ
        axes[1].imshow(cv_overlay_rgb)
        cv_title = f'CV Analysis\nLevel: {cv_results["overall_damage_level"]}, Regions: {len(cv_results["damage_regions"])}'
        axes[1].set_title(cv_title, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 3. ML анализ (если доступен)
        if ml_results and n_cols == 3:
            # Создаем визуализацию ML результатов
            ml_overlay = self._create_ml_overlay(original, ml_results)
            ml_overlay_rgb = cv2.cvtColor(ml_overlay, cv2.COLOR_BGR2RGB)
            
            axes[2].imshow(ml_overlay_rgb)
            ml_title = f'ML Analysis\nLevel {ml_results.get("damage_level", 0)}, Conf: {ml_results.get("confidence", 0):.2f}'
            axes[2].set_title(ml_title, fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        # Добавляем общую информацию
        damage_counts = cv_results['damage_counts']
        info_text = f"CV Results: Rust: {damage_counts['rust']}, Dents: {damage_counts['dents']}, Scratches: {damage_counts['scratches']}, Missing: {damage_counts['missing_parts']}"
        
        fig.suptitle('Damage Detection Analysis', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def _create_ml_overlay(self, image: np.ndarray, ml_results: Dict) -> np.ndarray:
        """
        Создание overlay для ML результатов
        
        Args:
            image: исходное изображение
            ml_results: результаты ML модели
            
        Returns:
            Изображение с ML overlay
        """
        overlay = image.copy()
        
        # Если есть bounding boxes от ML модели
        if 'bboxes' in ml_results:
            for bbox_info in ml_results['bboxes']:
                bbox = bbox_info['bbox']  # [x, y, w, h]
                confidence = bbox_info.get('confidence', 0.5)
                class_name = bbox_info.get('class', 'damage')
                
                x, y, w, h = bbox
                
                # Цвет в зависимости от уверенности
                if confidence > 0.8:
                    color = (0, 255, 0)      # Зеленый - высокая уверенность
                elif confidence > 0.5:
                    color = (0, 255, 255)    # Желтый - средняя уверенность
                else:
                    color = (0, 0, 255)      # Красный - низкая уверенность
                
                # Рисуем рамку
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Подпись
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(overlay, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Если нет конкретных bounding boxes, показываем общий результат
            damage_level = ml_results.get('damage_level', 0)
            confidence = ml_results.get('confidence', 0.0)
            
            # Цветовое покрытие в зависимости от уровня повреждения
            if damage_level == 0:
                color = (0, 255, 0)      # Зеленый - нет повреждений
            elif damage_level == 1:
                color = (0, 255, 255)    # Желтый - легкие повреждения
            elif damage_level == 2:
                color = (0, 165, 255)    # Оранжевый - средние повреждения
            else:
                color = (0, 0, 255)      # Красный - серьезные повреждения
            
            # Добавляем полупрозрачное покрытие
            if damage_level > 0:
                colored_overlay = overlay.copy()
                colored_overlay[:] = color
                overlay = cv2.addWeighted(overlay, 0.8, colored_overlay, 0.2, 0)
            
            # Текст с общей информацией
            h, w = overlay.shape[:2]
            text = f'ML: Level {damage_level}, Conf: {confidence:.2f}'
            cv2.putText(overlay, text, (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Добавляем типы повреждений если есть
            if 'damage_types' in ml_results:
                for i, damage_type in enumerate(ml_results['damage_types']):
                    cv2.putText(overlay, f'• {damage_type}', (10, h - 60 - i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay
    
    def create_damage_heatmap(self, image: np.ndarray, regions: List[DamageRegion]) -> np.ndarray:
        """
        Создание тепловой карты повреждений
        
        Args:
            image: исходное изображение
            regions: список областей повреждений
            
        Returns:
            Тепловая карта повреждений
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for region in regions:
            x, y, region_w, region_h = region.bbox
            
            # Вес основан на серьезности и уверенности
            weight = region.severity * region.confidence
            
            # Создаем гауссово распределение для региона
            center_x, center_y = x + region_w // 2, y + region_h // 2
            sigma_x, sigma_y = region_w // 4, region_h // 4
            
            # Генерируем координаты для региона
            y_coords, x_coords = np.ogrid[max(0, y):min(h, y + region_h), 
                                         max(0, x):min(w, x + region_w)]
            
            # Гауссово распределение
            gaussian = weight * np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) + 
                                        (y_coords - center_y)**2 / (2 * sigma_y**2)))
            
            # Добавляем к общей тепловой карте
            heatmap[max(0, y):min(h, y + region_h), 
                   max(0, x):min(w, x + region_w)] += gaussian
        
        # Нормализуем
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Применяем цветовую карту
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Накладываем на оригинал
        result = cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)
        
        return result
    
    def save_analysis_results(self, output_dir: Union[str, Path], filename_base: str,
                             original: np.ndarray, cv_results: Dict, 
                             ml_results: Optional[Dict] = None) -> Dict[str, str]:
        """
        Сохранение всех результатов анализа
        
        Args:
            output_dir: папка для сохранения
            filename_base: базовое имя файла
            original: оригинальное изображение
            cv_results: результаты CV анализа
            ml_results: результаты ML анализа
            
        Returns:
            Словарь с путями сохраненных файлов
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Сохраняем CV overlay
        cv_overlay = self.create_analysis_overlay(original, cv_results)
        cv_path = output_dir / f'{filename_base}_cv_analysis.jpg'
        cv2.imwrite(str(cv_path), cv_overlay)
        saved_files['cv_analysis'] = str(cv_path)
        
        # 2. Сохраняем тепловую карту
        heatmap = self.create_damage_heatmap(original, cv_results['damage_regions'])
        heatmap_path = output_dir / f'{filename_base}_heatmap.jpg'
        cv2.imwrite(str(heatmap_path), heatmap)
        saved_files['heatmap'] = str(heatmap_path)
        
        # 3. Сохраняем сводную визуализацию
        fig = self.create_summary_visualization(original, cv_results, ml_results)
        summary_path = output_dir / f'{filename_base}_summary.png'
        fig.savefig(str(summary_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files['summary'] = str(summary_path)
        
        # 4. Сохраняем маску повреждений
        mask = self.create_damage_mask(original, cv_results['damage_regions'])
        mask_path = output_dir / f'{filename_base}_damage_mask.jpg'
        cv2.imwrite(str(mask_path), mask)
        saved_files['damage_mask'] = str(mask_path)
        
        return saved_files
