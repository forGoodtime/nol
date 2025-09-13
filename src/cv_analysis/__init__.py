"""
Модуль компьютерного зрения для анализа повреждений автомобилей
"""

from .damage_detector import DamageDetectorCV
from .visual_analyzer import VisualAnalyzer
from .comparison_visualizer import ComparisonVisualizer

__all__ = ['DamageDetectorCV', 'VisualAnalyzer', 'ComparisonVisualizer']
