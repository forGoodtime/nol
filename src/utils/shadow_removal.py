"""
Утилиты для предварительной обработки изображений и удаления теней/бликов
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ShadowRemover:
    """Класс для удаления теней и бликов с изображений автомобилей"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Основной метод удаления теней (использует полный pipeline)
        
        Args:
            image: входное изображение в формате BGR или RGB
            
        Returns:
            обработанное изображение
        """
        return self.process_image_pipeline(image)
    
    def remove_shadows_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Удаление теней с помощью CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: входное изображение в формате BGR или RGB
            
        Returns:
            обработанное изображение
        """
        # Конвертируем в LAB цветовое пространство
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB if len(image.shape) == 3 else cv2.COLOR_BGR2LAB)
        
        # Применяем CLAHE к L каналу
        lab[:,:,0] = self.clahe.apply(lab[:,:,0])
        
        # Конвертируем обратно
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB if len(image.shape) == 3 else cv2.COLOR_LAB2BGR)
        
        return result
    
    def remove_shadows_morphological(self, image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        Удаление теней с помощью морфологических операций
        
        Args:
            image: входное изображение
            kernel_size: размер морфологического ядра
            
        Returns:
            обработанное изображение
        """
        # Конвертируем в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Создаем структурирующий элемент
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Морфологическое размыкание для удаления теней
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Создаем маску теней
        shadow_mask = cv2.absdiff(gray, opened)
        
        # Пороговая обработка
        _, shadow_mask = cv2.threshold(shadow_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Применяем маску к оригинальному изображению
        if len(image.shape) == 3:
            result = image.copy()
            for c in range(3):
                result[:,:,c] = cv2.bitwise_or(result[:,:,c], shadow_mask)
        else:
            result = cv2.bitwise_or(image, shadow_mask)
        
        return result
    
    def enhance_contrast_adaptive(self, image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """
        Адаптивное улучшение контраста
        
        Args:
            image: входное изображение
            alpha: коэффициент контраста (1.0-3.0)
            beta: коэффициент яркости (0-100)
            
        Returns:
            улучшенное изображение
        """
        # Базовое улучшение контраста
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Применяем билатеральный фильтр для сглаживания
        if len(enhanced.shape) == 3:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        else:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def remove_glare_highlights(self, image: np.ndarray, threshold: int = 240) -> np.ndarray:
        """
        Удаление бликов и пересвеченных областей
        
        Args:
            image: входное изображение
            threshold: порог для определения бликов
            
        Returns:
            изображение без бликов
        """
        if len(image.shape) == 3:
            # Конвертируем в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Находим яркие области (блики)
            glare_mask = cv2.inRange(hsv[:,:,2], threshold, 255)
            
            # Расширяем маску бликов
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_DILATE, kernel)
            
            # Применяем медианное размытие к областям с бликами
            result = image.copy()
            blurred = cv2.medianBlur(image, 7)
            
            # Заменяем блики размытой версией
            result[glare_mask > 0] = blurred[glare_mask > 0]
            
        else:
            # Для градаций серого
            glare_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
            result = image.copy()
            blurred = cv2.medianBlur(image, 7)
            result[glare_mask > 0] = blurred[glare_mask > 0]
        
        return result
    
    def white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Коррекция баланса белого для устранения цветовых искажений от освещения
        
        Args:
            image: входное изображение в формате RGB
            
        Returns:
            откорректированное изображение
        """
        if len(image.shape) != 3:
            return image
        
        # Простая коррекция баланса белого (Gray World Assumption)
        result = image.copy().astype(np.float64)
        
        # Вычисляем средние значения каналов
        mean_r = np.mean(result[:,:,0])
        mean_g = np.mean(result[:,:,1]) 
        mean_b = np.mean(result[:,:,2])
        
        # Общее среднее
        mean_gray = (mean_r + mean_g + mean_b) / 3
        
        # Коррекционные коэффициенты
        if mean_r > 0: result[:,:,0] *= mean_gray / mean_r
        if mean_g > 0: result[:,:,1] *= mean_gray / mean_g
        if mean_b > 0: result[:,:,2] *= mean_gray / mean_b
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process_image_pipeline(
        self, 
        image: Union[np.ndarray, Image.Image],
        remove_shadows: bool = True,
        remove_glare: bool = True,
        enhance_contrast: bool = True,
        white_balance: bool = True
    ) -> np.ndarray:
        """
        Полный пайплайн обработки изображения для борьбы с тенями и бликами
        
        Args:
            image: входное изображение
            remove_shadows: применять ли удаление теней
            remove_glare: применять ли удаление бликов
            enhance_contrast: применять ли улучшение контраста
            white_balance: применять ли коррекцию баланса белого
            
        Returns:
            обработанное изображение
        """
        # Конвертируем PIL в numpy если нужно
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        logger.info("Начинаем обработку изображения")
        
        try:
            # Коррекция баланса белого
            if white_balance:
                img_array = self.white_balance_correction(img_array)
                logger.debug("Применена коррекция баланса белого")
            
            # Удаление бликов
            if remove_glare:
                img_array = self.remove_glare_highlights(img_array)
                logger.debug("Удалены блики")
            
            # Удаление теней
            if remove_shadows:
                img_array = self.remove_shadows_clahe(img_array)
                logger.debug("Удалены тени с помощью CLAHE")
            
            # Улучшение контраста
            if enhance_contrast:
                img_array = self.enhance_contrast_adaptive(img_array)
                logger.debug("Улучшен контраст")
            
            logger.info("Обработка изображения завершена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            # Возвращаем оригинальное изображение в случае ошибки
            img_array = np.array(image) if isinstance(image, Image.Image) else image
        
        return img_array


def detect_shadow_regions(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Детекция областей с тенями на изображении
    
    Args:
        image: входное изображение
        threshold: порог для определения теней (0.0-1.0)
        
    Returns:
        бинарная маска теней
    """
    if len(image.shape) == 3:
        # Конвертируем в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Низкая яркость и низкая насыщенность указывают на тени
        low_value = hsv[:,:,2] < (threshold * 255)
        low_saturation = hsv[:,:,1] < (0.5 * 255)
        
        shadow_mask = low_value & low_saturation
    else:
        # Для градаций серого - просто низкая яркость
        shadow_mask = image < (threshold * 255)
    
    # Морфологическая обработка для очистки маски
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return shadow_mask


def is_shadow_or_defect(roi: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Определяет, является ли область тенью или дефектом
    
    Args:
        roi: область интереса (Region of Interest)
        confidence_threshold: порог уверенности
        
    Returns:
        (is_shadow, confidence) - True если тень, False если дефект
    """
    if len(roi.shape) == 3:
        # Анализируем цветовые характеристики
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Средние значения HSV
        mean_h = np.mean(hsv_roi[:,:,0])
        mean_s = np.mean(hsv_roi[:,:,1])
        mean_v = np.mean(hsv_roi[:,:,2])
        
        # Стандартные отклонения
        std_h = np.std(hsv_roi[:,:,0])
        std_s = np.std(hsv_roi[:,:,1])
        std_v = np.std(hsv_roi[:,:,2])
        
        # Признаки тени:
        # 1. Низкая яркость (V)
        # 2. Низкая вариация цвета (H)
        # 3. Низкая насыщенность (S)
        
        shadow_score = 0.0
        
        # Низкая яркость
        if mean_v < 100:  # из 255
            shadow_score += 0.4
        
        # Низкая насыщенность
        if mean_s < 50:   # из 255
            shadow_score += 0.3
            
        # Низкая вариация цвета
        if std_h < 10:    # из 180
            shadow_score += 0.3
        
        is_shadow = shadow_score >= confidence_threshold
        return is_shadow, shadow_score
    
    else:
        # Для градаций серого - анализ текстуры
        # Тени обычно имеют более однородную текстуру
        
        # Вычисляем локальную бинарную текстуру (упрощенно)
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Низкая вариация текстуры + низкая яркость = тень
        mean_intensity = np.mean(roi)
        
        is_shadow = (mean_intensity < 80) and (texture_variance < 100)
        confidence = 0.8 if is_shadow else 0.2
        
        return is_shadow, confidence
