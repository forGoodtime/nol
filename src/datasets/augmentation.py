"""
Аугментации для обучающих и валидационных данных
"""
from typing import Callable, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_transforms(image_size: tuple = (640, 640), config: dict = None) -> Callable:
    """
    Возвращает трансформации для обучающих данных
    
    Args:
        image_size: размер изображения (height, width)
        config: конфигурация аугментаций
    """
    if config is None:
        config = {}
    
    transforms = [
        # Изменение размера
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Геометрические трансформации
        A.HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=config.get('vertical_flip', 0.1)),
        A.Rotate(
            limit=config.get('rotation', 15), 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=config.get('translate', 0.1),
            scale_limit=config.get('scale', 0.2),
            rotate_limit=config.get('rotation', 15),
            border_mode=cv2.BORDER_CONSTANT,
            value=0, p=0.5
        ),
        
        # Цветовые трансформации (критичны для теней/бликов)
        A.RandomBrightnessContrast(
            brightness_limit=config.get('brightness', 0.2),
            contrast_limit=config.get('contrast', 0.2),
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=config.get('hue', 0.1),
            sat_shift_limit=config.get('saturation', 0.2),
            val_shift_limit=config.get('brightness', 0.2),
            p=0.5
        ),
        
        # CLAHE для улучшения контраста (борьба с тенями)
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.5 if config.get('clahe', True) else 0.0
        ),
        
        # Размытие (имитация камеры)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=config.get('gaussian_blur', 0.3)),
        
        # Шум
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Искажения линз
        A.OpticalDistortion(
            distort_limit=0.1, 
            shift_limit=0.1, 
            p=0.2
        ),
        
        # Погодные условия
        A.OneOf([
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=20, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7, brightness_coefficient=0.7,
                rain_type='drizzle', p=1.0
            ),
            A.RandomFog(
                fog_coeff_lower=0.1, fog_coeff_upper=0.3,
                alpha_coeff=0.08, p=1.0
            ),
        ], p=0.1),
    ]
    
    # Специальные аугментации для автомобилей
    if config.get('random_shadow', 0.0) > 0:
        transforms.append(RandomShadow(p=config['random_shadow']))
    
    if config.get('random_brightness', 0.0) > 0:
        transforms.append(A.RandomBrightness(limit=0.2, p=config['random_brightness']))
    
    # Финальные трансформации
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def get_val_transforms(image_size: tuple = (640, 640)) -> Callable:
    """Трансформации для валидационных данных (без аугментаций)"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_test_transforms(image_size: tuple = (640, 640)) -> Callable:
    """Трансформации для тестовых данных"""
    return get_val_transforms(image_size)


class RandomShadow(A.ImageOnlyTransform):
    """Кастомная аугментация для добавления теней"""
    
    def __init__(self, shadow_roi=(0.1, 0.1, 0.9, 0.9), shadow_intensity=(0.3, 0.7), 
                 always_apply=False, p=0.5):
        super(RandomShadow, self).__init__(always_apply, p)
        self.shadow_roi = shadow_roi
        self.shadow_intensity = shadow_intensity
    
    def apply(self, img, **params):
        h, w = img.shape[:2]
        
        # Определяем область для тени
        x1 = int(self.shadow_roi[0] * w)
        y1 = int(self.shadow_roi[1] * h)
        x2 = int(self.shadow_roi[2] * w)
        y2 = int(self.shadow_roi[3] * h)
        
        # Случайная интенсивность тени
        intensity = np.random.uniform(*self.shadow_intensity)
        
        # Создаем маску тени (градиентную)
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # Случайная форма тени
        shadow_type = np.random.choice(['rectangle', 'ellipse', 'polygon'])
        
        if shadow_type == 'rectangle':
            shadow_mask[y1:y2, x1:x2] = intensity
        elif shadow_type == 'ellipse':
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            a = (x2 - x1) // 2
            b = (y2 - y1) // 2
            
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2) / (a ** 2) + ((y - center_y) ** 2) / (b ** 2) <= 1
            shadow_mask[mask] = intensity
        
        # Применяем тень
        img_shadow = img.copy().astype(np.float32)
        for c in range(3):
            img_shadow[:, :, c] *= shadow_mask
        
        return np.clip(img_shadow, 0, 255).astype(np.uint8)


def get_segmentation_transforms(image_size: tuple = (640, 640), is_training: bool = True) -> Callable:
    """Трансформации для задач сегментации (с масками)"""
    if is_training:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
