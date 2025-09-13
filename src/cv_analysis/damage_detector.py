"""
Детектор повреждений на основе компьютерного зрения
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DamageRegion:
    """Класс для представления области повреждения"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    damage_type: str
    confidence: float
    severity: int  # 0-3 (нет, легкие, средние, серьезные)
    area: int
    contour: Optional[np.ndarray] = None


class DamageDetectorCV:
    """
    Детектор повреждений автомобилей на основе компьютерного зрения
    Использует традиционные методы CV для обнаружения:
    - Ржавчины (коричневые/оранжевые области)
    - Вмятин (изменения в геометрии/освещении)
    - Царапин (линейные дефекты)
    - Недостающих частей (резкие контуры, пустые области)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Инициализируем детекторы
        self._init_color_ranges()
        self._init_morphology_kernels()
        
        logger.info("DamageDetectorCV инициализирован")
    
    def _get_default_config(self) -> Dict:
        """Конфигурация по умолчанию"""
        return {
            'rust_detection': {
                'hsv_lower': [10, 50, 50],    # Нижний порог для ржавчины в HSV
                'hsv_upper': [25, 255, 255],  # Верхний порог для ржавчины в HSV
                'min_area': 100,              # Минимальная площадь области
                'blur_kernel': 5,             # Размер ядра для размытия
            },
            'dent_detection': {
                'sobel_threshold': 50,        # Порог для детекции градиентов
                'contour_min_area': 200,      # Минимальная площадь контура
                'aspect_ratio_range': [0.3, 3.0],  # Диапазон соотношений сторон
            },
            'scratch_detection': {
                'line_threshold': 30,         # Порог для детекции линий
                'min_line_length': 20,        # Минимальная длина линии
                'max_line_gap': 5,            # Максимальный разрыв в линии
            },
            'missing_parts': {
                'edge_threshold': 100,        # Порог для детекции краев
                'hole_min_area': 500,         # Минимальная площадь "дыры"
            },
            'general': {
                'confidence_threshold': 0.3,  # Общий порог уверенности
                'nms_threshold': 0.4,         # Non-maximum suppression
            }
        }
    
    def _init_color_ranges(self):
        """Инициализация цветовых диапазонов для детекции"""
        # Ржавчина: оранжево-коричневые оттенки
        self.rust_hsv_lower = np.array(self.config['rust_detection']['hsv_lower'])
        self.rust_hsv_upper = np.array(self.config['rust_detection']['hsv_upper'])
        
        # Дополнительные диапазоны для разных типов ржавчины
        self.rust_ranges = [
            (np.array([10, 50, 50]), np.array([25, 255, 255])),    # Основная ржавчина
            (np.array([0, 100, 50]), np.array([10, 255, 200])),    # Темная ржавчина
            (np.array([15, 30, 80]), np.array([30, 150, 255]))     # Светлая ржавчина
        ]
    
    def _init_morphology_kernels(self):
        """Инициализация морфологических ядер"""
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Ядро для детекции линий (царапин)
        self.line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        self.line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        self.line_kernel_d1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        self.line_kernel_d2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    
    def detect_rust(self, image: np.ndarray) -> List[DamageRegion]:
        """
        Детекция ржавчины на основе цветового анализа
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Применяем размытие для уменьшения шума
        blur_kernel = self.config['rust_detection']['blur_kernel']
        hsv_blurred = cv2.GaussianBlur(hsv, (blur_kernel, blur_kernel), 0)
        
        rust_regions = []
        
        # Проверяем каждый цветовой диапазон ржавчины
        for i, (lower, upper) in enumerate(self.rust_ranges):
            mask = cv2.inRange(hsv_blurred, lower, upper)
            
            # Морфологические операции для очистки маски
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
            
            # Находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config['rust_detection']['min_area']:
                    continue
                
                # Вычисляем bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Вычисляем уверенность на основе площади и интенсивности
                roi = hsv_blurred[y:y+h, x:x+w]
                mean_sat = np.mean(roi[:, :, 1])  # Средняя насыщенность
                confidence = min(0.9, (area / 1000.0) * (mean_sat / 255.0) + 0.3)
                
                # Определяем серьезность на основе площади
                if area < 500:
                    severity = 1  # Легкая ржавчина
                elif area < 2000:
                    severity = 2  # Средняя ржавчина
                else:
                    severity = 3  # Серьезная ржавчина
                
                rust_regions.append(DamageRegion(
                    bbox=(x, y, w, h),
                    damage_type=f'rust_type_{i+1}',
                    confidence=confidence,
                    severity=severity,
                    area=int(area),
                    contour=contour
                ))
        
        return rust_regions
    
    def detect_dents(self, image: np.ndarray) -> List[DamageRegion]:
        """
        Детекция вмятин на основе анализа градиентов и теней
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем Gaussian blur для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Вычисляем градиенты Sobel
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Магнитуда градиента
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Нормализуем и применяем порог
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        _, threshold = cv2.threshold(magnitude, self.config['dent_detection']['sobel_threshold'], 255, cv2.THRESH_BINARY)
        
        # Морфологические операции
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.kernel_medium)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, self.kernel_small)
        
        # Находим контуры
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dent_regions = []
        min_area = self.config['dent_detection']['contour_min_area']
        aspect_range = self.config['dent_detection']['aspect_ratio_range']
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Проверяем соотношение сторон
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_range[0] <= aspect_ratio <= aspect_range[1]:
                # Анализируем интенсивность в области
                roi = gray[y:y+h, x:x+w]
                std_intensity = np.std(roi)
                
                # Уверенность основана на вариации интенсивности и площади
                confidence = min(0.8, (std_intensity / 50.0) * (area / 1000.0) + 0.2)
                
                # Серьезность на основе площади и вариации
                if area < 1000 or std_intensity < 20:
                    severity = 1
                elif area < 3000 or std_intensity < 40:
                    severity = 2
                else:
                    severity = 3
                
                dent_regions.append(DamageRegion(
                    bbox=(x, y, w, h),
                    damage_type='dent',
                    confidence=confidence,
                    severity=severity,
                    area=int(area),
                    contour=contour
                ))
        
        return dent_regions
    
    def detect_scratches(self, image: np.ndarray) -> List[DamageRegion]:
        """
        Детекция царапин на основе детекции линий
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем адаптивное размытие
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Детекция краев Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Детекция линий с помощью HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=self.config['scratch_detection']['line_threshold'],
            minLineLength=self.config['scratch_detection']['min_line_length'],
            maxLineGap=self.config['scratch_detection']['max_line_gap']
        )
        
        scratch_regions = []
        
        if lines is not None:
            # Группируем близкие линии
            line_groups = self._group_lines(lines)
            
            for group in line_groups:
                # Вычисляем bounding box для группы линий
                all_points = np.array([[[line[0], line[1]], [line[2], line[3]]] for line in group]).reshape(-1, 2)
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
                
                x, y = int(x_min), int(y_min)
                w, h = int(x_max - x_min), int(y_max - y_min)
                
                # Добавляем отступ
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                
                # Вычисляем уверенность на основе количества и длины линий
                total_length = sum([np.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2) for line in group])
                confidence = min(0.9, len(group) * 0.1 + (total_length / 1000.0))
                
                # Серьезность на основе общей длины и количества линий
                if total_length < 50 or len(group) < 3:
                    severity = 1
                elif total_length < 200 or len(group) < 8:
                    severity = 2
                else:
                    severity = 3
                
                scratch_regions.append(DamageRegion(
                    bbox=(x, y, w, h),
                    damage_type='scratch',
                    confidence=confidence,
                    severity=severity,
                    area=w * h,
                    contour=None
                ))
        
        return scratch_regions
    
    def detect_missing_parts(self, image: np.ndarray) -> List[DamageRegion]:
        """
        Детекция недостающих частей на основе анализа контуров и "дыр"
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Детекция краев
        edges = cv2.Canny(gray, self.config['missing_parts']['edge_threshold'], 
                         self.config['missing_parts']['edge_threshold'] * 2)
        
        # Морфологические операции для замыкания контуров
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel_large)
        
        # Заполнение дыр
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем маску для заполнения
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Инвертируем для поиска дыр
        holes_mask = cv2.bitwise_not(mask)
        
        # Находим контуры дыр
        hole_contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        missing_regions = []
        min_area = self.config['missing_parts']['hole_min_area']
        
        for contour in hole_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Проверяем, что это действительно "дыра", а не край изображения
            x, y, w, h = cv2.boundingRect(contour)
            
            # Исключаем области на краях изображения
            if (x < 10 or y < 10 or 
                x + w > image.shape[1] - 10 or 
                y + h > image.shape[0] - 10):
                continue
            
            # Анализируем окружение дыры
            roi = gray[max(0, y-20):min(gray.shape[0], y+h+20), 
                      max(0, x-20):min(gray.shape[1], x+w+20)]
            
            # Если окружение имеет высокую вариацию, вероятно это повреждение
            std_around = np.std(roi)
            
            confidence = min(0.9, (area / 2000.0) + (std_around / 100.0))
            
            # Серьезность всегда высокая для недостающих частей
            severity = 3
            
            missing_regions.append(DamageRegion(
                bbox=(x, y, w, h),
                damage_type='missing_part',
                confidence=confidence,
                severity=severity,
                area=int(area),
                contour=contour
            ))
        
        return missing_regions
    
    def _group_lines(self, lines: np.ndarray, distance_threshold: float = 20.0, 
                    angle_threshold: float = 10.0) -> List[List]:
        """Группировка близких линий"""
        if lines is None or len(lines) == 0:
            return []
        
        lines = lines.reshape(-1, 4)
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used or i == j:
                    continue
                
                # Вычисляем расстояние и угол между линиями
                if self._lines_are_similar(line1, line2, distance_threshold, angle_threshold):
                    group.append(line2)
                    used.add(j)
            
            if len(group) >= 2:  # Группы из одной линии игнорируем
                groups.append(group)
        
        return groups
    
    def _lines_are_similar(self, line1: np.ndarray, line2: np.ndarray, 
                          distance_threshold: float, angle_threshold: float) -> bool:
        """Проверка схожести двух линий"""
        # Вычисляем углы линий
        angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0]) * 180 / np.pi
        angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0]) * 180 / np.pi
        
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff > angle_threshold:
            return False
        
        # Вычисляем расстояние между центрами линий
        center1 = [(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2]
        center2 = [(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2]
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance <= distance_threshold
    
    def analyze_image(self, image: Union[np.ndarray, str, Path]) -> Dict:
        """
        Полный анализ изображения на предмет повреждений
        
        Args:
            image: изображение (массив numpy, путь к файлу или Path объект)
            
        Returns:
            Словарь с результатами анализа
        """
        # Загружаем изображение если передан путь
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
        
        # Применяем все детекторы
        rust_regions = self.detect_rust(image)
        dent_regions = self.detect_dents(image)
        scratch_regions = self.detect_scratches(image)
        missing_regions = self.detect_missing_parts(image)
        
        # Объединяем все области
        all_regions = rust_regions + dent_regions + scratch_regions + missing_regions
        
        # Применяем Non-Maximum Suppression для удаления перекрывающихся регионов
        filtered_regions = self._apply_nms(all_regions)
        
        # Определяем общий уровень повреждения
        overall_damage_level = self._calculate_overall_damage_level(filtered_regions)
        
        return {
            'damage_regions': filtered_regions,
            'damage_counts': {
                'rust': len([r for r in filtered_regions if 'rust' in r.damage_type]),
                'dents': len([r for r in filtered_regions if r.damage_type == 'dent']),
                'scratches': len([r for r in filtered_regions if r.damage_type == 'scratch']),
                'missing_parts': len([r for r in filtered_regions if r.damage_type == 'missing_part']),
            },
            'overall_damage_level': overall_damage_level,
            'total_damaged_area': sum([r.area for r in filtered_regions]),
            'image_size': image.shape[:2],
            'confidence_scores': [r.confidence for r in filtered_regions]
        }
    
    def _apply_nms(self, regions: List[DamageRegion]) -> List[DamageRegion]:
        """Применение Non-Maximum Suppression"""
        if not regions:
            return []
        
        # Фильтруем по минимальной уверенности
        min_confidence = self.config['general']['confidence_threshold']
        regions = [r for r in regions if r.confidence >= min_confidence]
        
        if not regions:
            return []
        
        # Сортируем по уверенности (убывание)
        regions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Применяем NMS
        keep = []
        while regions:
            current = regions.pop(0)
            keep.append(current)
            
            # Удаляем регионы с высоким IoU
            regions = [r for r in regions if self._calculate_iou(current, r) < self.config['general']['nms_threshold']]
        
        return keep
    
    def _calculate_iou(self, region1: DamageRegion, region2: DamageRegion) -> float:
        """Вычисление Intersection over Union для двух регионов"""
        x1, y1, w1, h1 = region1.bbox
        x2, y2, w2, h2 = region2.bbox
        
        # Координаты пересечения
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        # Площади
        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _calculate_overall_damage_level(self, regions: List[DamageRegion]) -> int:
        """Вычисление общего уровня повреждения"""
        if not regions:
            return 0
        
        # Учитываем максимальную серьезность и количество повреждений
        max_severity = max([r.severity for r in regions])
        total_regions = len(regions)
        
        # Логика определения общего уровня
        if max_severity == 3 or total_regions > 10:
            return 3  # Серьезные повреждения
        elif max_severity == 2 or total_regions > 5:
            return 2  # Средние повреждения
        elif max_severity == 1 or total_regions > 0:
            return 1  # Легкие повреждения
        else:
            return 0  # Нет повреждений
