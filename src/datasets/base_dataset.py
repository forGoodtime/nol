"""
Базовый класс для датасетов
"""
import os
from typing import Optional, Callable, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """Базовый класс для всех датасетов в проекте"""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir: путь к корневой папке с данными
            transform: трансформации для изображений
            target_transform: трансформации для таргетов
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Список файлов будет определен в наследниках
        self.samples = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Базовая реализация загрузки элемента
        Должна быть переопределена в наследниках
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def load_image(self, image_path: str) -> Image.Image:
        """Загружает изображение из файла"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {image_path}: {e}")
    
    def validate_image(self, image: Image.Image) -> bool:
        """Проверяет валидность изображения"""
        if image.size[0] < 32 or image.size[1] < 32:
            return False
        return True
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Вычисляет веса классов для балансировки
        Должна быть переопределена в наследниках
        """
        return torch.ones(4, dtype=torch.float32)  # по умолчанию равные веса
    
    def get_sample_info(self, idx: int) -> dict:
        """Возвращает информацию о сэмпле (для дебага)"""
        return {
            "index": idx,
            "total_samples": len(self),
            "sample_path": self.samples[idx] if idx < len(self.samples) else None
        }
