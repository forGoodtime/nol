"""
Модули для работы с данными
"""

from .base_dataset import BaseDataset
from .defect_coco_dataset import DefectCocoDataset  
from .classification_dataset import ClassificationDataset
from .augmentation import get_train_transforms, get_val_transforms
