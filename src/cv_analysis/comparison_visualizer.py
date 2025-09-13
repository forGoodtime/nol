"""
Компаратор для совместного анализа CV и ML результатов
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import json
from pathlib import Path
import logging

from .damage_detector import DamageRegion, DamageDetectorCV
from .visual_analyzer import VisualAnalyzer

logger = logging.getLogger(__name__)


class ComparisonVisualizer:
    """
    Класс для сравнения и визуализации результатов CV и ML анализа
    """
    
    def __init__(self):
        self.cv_detector = DamageDetectorCV()
        self.visual_analyzer = VisualAnalyzer()
        
        # Маппинг уровней повреждений
        self.damage_level_names = {
            0: "Нет повреждений",
            1: "Легкие повреждения",
            2: "Средние повреждения", 
            3: "Серьезные повреждения"
        }
        
    def analyze_with_both_methods(self, image: Union[np.ndarray, str, Path],
                                 ml_model_path: Optional[str] = None) -> Dict:
        """
        Анализ изображения с помощью CV и ML методов
        
        Args:
            image: изображение для анализа
            ml_model_path: путь к ML модели (опционально)
            
        Returns:
            Объединенные результаты CV и ML анализа
        """
        # Загружаем изображение если необходимо
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
        else:
            img_array = image.copy()
        
        # CV анализ
        logger.info("Запуск CV анализа...")
        cv_results = self.cv_detector.analyze_image(img_array)
        
        # ML анализ (мокируем если модель недоступна)
        logger.info("Запуск ML анализа...")
        if ml_model_path and Path(ml_model_path).exists():
            ml_results = self._run_ml_analysis(img_array, ml_model_path)
        else:
            ml_results = self._mock_ml_analysis(img_array)
        
        # Сравнение результатов
        comparison = self._compare_results(cv_results, ml_results)
        
        return {
            'cv_results': cv_results,
            'ml_results': ml_results,
            'comparison': comparison,
            'image_shape': img_array.shape
        }
    
    def _run_ml_analysis(self, image: np.ndarray, model_path: str) -> Dict:
        """
        Запуск ML анализа с реальной моделью
        """
        # Здесь будет интеграция с реальной ML моделью
        # Пока используем заглушку
        return self._mock_ml_analysis(image)
    
    def _mock_ml_analysis(self, image: np.ndarray) -> Dict:
        """
        Имитация ML анализа для демонстрации
        """
        # Простая эвристика для имитации ML результатов
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Анализ характеристик изображения
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Логика имитации
        if std_intensity < 15 and mean_intensity > 100:
            damage_level = 0
            confidence = 0.92
            damage_types = []
        elif laplacian_var > 800:  # Высокая вариация = много деталей/повреждений
            damage_level = 3
            confidence = 0.87
            damage_types = ["severe_damage", "missing_part"]
        elif mean_intensity < 80:  # Темные области
            damage_level = 2
            confidence = 0.75
            damage_types = ["rust", "dent"]
        else:
            damage_level = 1
            confidence = 0.68
            damage_types = ["light_scratches"]
        
        # Генерируем фиктивные bounding boxes
        h, w = image.shape[:2]
        bboxes = []
        
        if damage_level > 0:
            # Генерируем 1-3 случайных bbox'а
            n_boxes = min(damage_level, 3)
            for i in range(n_boxes):
                x = np.random.randint(w // 4, 3 * w // 4)
                y = np.random.randint(h // 4, 3 * h // 4)
                box_w = np.random.randint(50, min(200, w - x))
                box_h = np.random.randint(50, min(200, h - y))
                
                bboxes.append({
                    'bbox': [x, y, box_w, box_h],
                    'confidence': confidence + np.random.uniform(-0.1, 0.1),
                    'class': damage_types[i % len(damage_types)] if damage_types else 'damage'
                })
        
        return {
            'damage_level': damage_level,
            'confidence': confidence,
            'damage_types': damage_types,
            'bboxes': bboxes,
            'processing_time': np.random.uniform(0.05, 0.2)
        }
    
    def _compare_results(self, cv_results: Dict, ml_results: Dict) -> Dict:
        """
        Сравнение результатов CV и ML анализа
        """
        cv_level = cv_results['overall_damage_level']
        ml_level = ml_results['damage_level']
        
        # Согласованность уровней повреждений
        level_agreement = cv_level == ml_level
        level_difference = abs(cv_level - ml_level)
        
        # Сравнение количества обнаруженных областей
        cv_regions_count = len(cv_results['damage_regions'])
        ml_regions_count = len(ml_results.get('bboxes', []))
        
        # Анализ типов повреждений
        cv_types = set()
        for region in cv_results['damage_regions']:
            damage_type = region.damage_type
            if 'rust' in damage_type:
                cv_types.add('rust')
            else:
                cv_types.add(damage_type)
        
        ml_types = set(ml_results.get('damage_types', []))
        
        # Пересечение типов повреждений
        common_types = cv_types.intersection(ml_types)
        type_agreement = len(common_types) / max(len(cv_types.union(ml_types)), 1)
        
        # Общая оценка согласованности
        if level_agreement and type_agreement > 0.5:
            agreement_status = "high"
        elif level_difference <= 1 and type_agreement > 0.3:
            agreement_status = "medium"
        else:
            agreement_status = "low"
        
        return {
            'level_agreement': level_agreement,
            'level_difference': level_difference,
            'agreement_status': agreement_status,
            'cv_regions_count': cv_regions_count,
            'ml_regions_count': ml_regions_count,
            'cv_damage_types': list(cv_types),
            'ml_damage_types': list(ml_types),
            'common_damage_types': list(common_types),
            'type_agreement_score': type_agreement,
            'recommendations': self._generate_recommendations(cv_results, ml_results, agreement_status)
        }
    
    def _generate_recommendations(self, cv_results: Dict, ml_results: Dict, 
                                agreement_status: str) -> List[str]:
        """
        Генерация рекомендаций на основе сравнения результатов
        """
        recommendations = []
        
        cv_level = cv_results['overall_damage_level']
        ml_level = ml_results['damage_level']
        cv_conf = np.mean(cv_results.get('confidence_scores', [0.5]))
        ml_conf = ml_results.get('confidence', 0.5)
        
        if agreement_status == "high":
            recommendations.append("✅ CV и ML методы показывают согласованные результаты")
            
            if cv_level > 2:
                recommendations.append("⚠️ Обнаружены серьезные повреждения - требуется детальный осмотр")
            elif cv_level > 0:
                recommendations.append("💡 Обнаружены повреждения - рекомендуется профилактика")
        
        elif agreement_status == "medium":
            recommendations.append("⚠️ CV и ML показывают частично согласованные результаты")
            
            if abs(cv_level - ml_level) == 1:
                recommendations.append("📊 Небольшое расхождение в оценке серьезности")
            
            if cv_conf < 0.6 or ml_conf < 0.6:
                recommendations.append("🔍 Низкая уверенность - требуется дополнительная проверка")
        
        else:  # low agreement
            recommendations.append("❌ Значительное расхождение между CV и ML результатами")
            recommendations.append("🔬 Рекомендуется ручная проверка и дополнительный анализ")
            
            if cv_level > ml_level + 1:
                recommendations.append("🔍 CV метод обнаружил больше повреждений чем ML")
            elif ml_level > cv_level + 1:
                recommendations.append("🤖 ML модель определила более высокий уровень повреждений")
        
        # Дополнительные рекомендации
        if len(cv_results['damage_regions']) > 10:
            recommendations.append("⚠️ Обнаружено множество областей повреждений")
        
        if cv_results['total_damaged_area'] > cv_results['image_size'][0] * cv_results['image_size'][1] * 0.1:
            recommendations.append("📏 Большая площадь повреждений относительно размера изображения")
        
        return recommendations
    
    def create_side_by_side_comparison(self, image: np.ndarray, 
                                      combined_results: Dict) -> Figure:
        """
        Создание сравнительной визуализации CV и ML результатов
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # Конвертируем изображение в RGB
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cv_results = combined_results['cv_results']
        ml_results = combined_results['ml_results']
        comparison = combined_results['comparison']
        
        # 1. Оригинальное изображение
        axes[0].imshow(original_rgb)
        axes[0].set_title('Оригинал', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 2. CV анализ
        cv_overlay = self.visual_analyzer.create_analysis_overlay(image, cv_results)
        cv_overlay_rgb = cv2.cvtColor(cv_overlay, cv2.COLOR_BGR2RGB)
        axes[1].imshow(cv_overlay_rgb)
        
        cv_title = f'Computer Vision\nLevel: {cv_results["overall_damage_level"]} | Regions: {len(cv_results["damage_regions"])}'
        axes[1].set_title(cv_title, fontsize=16, fontweight='bold', color='blue')
        axes[1].axis('off')
        
        # 3. ML анализ
        ml_overlay = self.visual_analyzer._create_ml_overlay(image, ml_results)
        ml_overlay_rgb = cv2.cvtColor(ml_overlay, cv2.COLOR_BGR2RGB)
        axes[2].imshow(ml_overlay_rgb)
        
        ml_title = f'Machine Learning\nLevel: {ml_results["damage_level"]} | Conf: {ml_results["confidence"]:.2f}'
        axes[2].set_title(ml_title, fontsize=16, fontweight='bold', color='green')
        axes[2].axis('off')
        
        # Общий заголовок с информацией о согласованности
        agreement_color = {'high': 'green', 'medium': 'orange', 'low': 'red'}
        agreement_text = f'Agreement: {comparison["agreement_status"].upper()}'
        
        fig.suptitle(f'CV vs ML Damage Detection Analysis | {agreement_text}', 
                    fontsize=20, fontweight='bold', 
                    color=agreement_color[comparison['agreement_status']])
        
        # Информационный текст внизу
        info_lines = [
            f"CV: {comparison['cv_regions_count']} regions, Types: {', '.join(comparison['cv_damage_types'])}",
            f"ML: {comparison['ml_regions_count']} regions, Types: {', '.join(comparison['ml_damage_types'])}",
            f"Common types: {', '.join(comparison['common_damage_types']) if comparison['common_damage_types'] else 'None'}"
        ]
        
        info_text = ' | '.join(info_lines)
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.12)
        
        return fig
    
    def create_detailed_report(self, combined_results: Dict) -> str:
        """
        Создание детального текстового отчета
        """
        cv_results = combined_results['cv_results']
        ml_results = combined_results['ml_results']
        comparison = combined_results['comparison']
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ДЕТАЛЬНЫЙ ОТЧЕТ: СРАВНЕНИЕ CV И ML АНАЛИЗА")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Общие результаты
        report_lines.append("ОБЩИЕ РЕЗУЛЬТАТЫ:")
        report_lines.append(f"  CV уровень повреждений: {cv_results['overall_damage_level']} - {self.damage_level_names[cv_results['overall_damage_level']]}")
        report_lines.append(f"  ML уровень повреждений: {ml_results['damage_level']} - {self.damage_level_names[ml_results['damage_level']]}")
        report_lines.append(f"  Согласованность: {comparison['agreement_status'].upper()}")
        report_lines.append("")
        
        # CV детали
        report_lines.append("COMPUTER VISION АНАЛИЗ:")
        report_lines.append(f"  Обнаружено регионов: {len(cv_results['damage_regions'])}")
        report_lines.append(f"  Общая поврежденная площадь: {cv_results['total_damaged_area']} пикселей")
        
        damage_counts = cv_results['damage_counts']
        report_lines.append("  Распределение по типам:")
        report_lines.append(f"    • Ржавчина: {damage_counts['rust']}")
        report_lines.append(f"    • Вмятины: {damage_counts['dents']}")
        report_lines.append(f"    • Царапины: {damage_counts['scratches']}")
        report_lines.append(f"    • Недостающие части: {damage_counts['missing_parts']}")
        
        if cv_results['confidence_scores']:
            avg_conf = np.mean(cv_results['confidence_scores'])
            report_lines.append(f"  Средняя уверенность: {avg_conf:.3f}")
        report_lines.append("")
        
        # ML детали
        report_lines.append("MACHINE LEARNING АНАЛИЗ:")
        report_lines.append(f"  Уверенность: {ml_results['confidence']:.3f}")
        report_lines.append(f"  Время обработки: {ml_results.get('processing_time', 0):.3f} сек")
        report_lines.append(f"  Обнаруженные типы: {', '.join(ml_results.get('damage_types', []))}")
        
        if ml_results.get('bboxes'):
            report_lines.append(f"  Количество bounding boxes: {len(ml_results['bboxes'])}")
            for i, bbox in enumerate(ml_results['bboxes'][:3]):  # Показываем первые 3
                report_lines.append(f"    Box {i+1}: {bbox['class']} (conf: {bbox['confidence']:.3f})")
        report_lines.append("")
        
        # Сравнение
        report_lines.append("АНАЛИЗ СОГЛАСОВАННОСТИ:")
        report_lines.append(f"  Совпадение уровней: {'ДА' if comparison['level_agreement'] else 'НЕТ'}")
        report_lines.append(f"  Разность уровней: {comparison['level_difference']}")
        report_lines.append(f"  Согласованность типов: {comparison['type_agreement_score']:.3f}")
        report_lines.append(f"  Общие типы повреждений: {', '.join(comparison['common_damage_types']) if comparison['common_damage_types'] else 'Нет'}")
        report_lines.append("")
        
        # Рекомендации
        report_lines.append("РЕКОМЕНДАЦИИ:")
        for rec in comparison['recommendations']:
            report_lines.append(f"  {rec}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\\n".join(report_lines)
    
    def save_comparison_results(self, output_dir: Union[str, Path], 
                              filename_base: str, image: np.ndarray,
                              combined_results: Dict) -> Dict[str, str]:
        """
        Сохранение всех результатов сравнительного анализа
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Сравнительная визуализация
        comparison_fig = self.create_side_by_side_comparison(image, combined_results)
        comparison_path = output_dir / f'{filename_base}_cv_ml_comparison.png'
        comparison_fig.savefig(str(comparison_path), dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        saved_files['comparison'] = str(comparison_path)
        
        # 2. Детальный отчет
        report = self.create_detailed_report(combined_results)
        report_path = output_dir / f'{filename_base}_detailed_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        saved_files['report'] = str(report_path)
        
        # 3. JSON с результатами
        json_path = output_dir / f'{filename_base}_results.json'
        
        # Подготавливаем данные для JSON (убираем numpy объекты)
        json_data = self._prepare_for_json(combined_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        saved_files['json'] = str(json_path)
        
        # 4. Индивидуальные визуализации CV
        cv_files = self.visual_analyzer.save_analysis_results(
            output_dir, f'{filename_base}_cv', image, 
            combined_results['cv_results'], combined_results['ml_results']
        )
        saved_files.update(cv_files)
        
        return saved_files
    
    def _prepare_for_json(self, data: Dict) -> Dict:
        """
        Подготовка данных для сериализации в JSON
        """
        def convert_item(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.integer):
                return int(item)
            elif isinstance(item, np.floating):
                return float(item)
            elif isinstance(item, DamageRegion):
                return {
                    'bbox': item.bbox,
                    'damage_type': item.damage_type,
                    'confidence': float(item.confidence),
                    'severity': int(item.severity),
                    'area': int(item.area)
                }
            elif isinstance(item, dict):
                return {key: convert_item(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [convert_item(x) for x in item]
            else:
                return item
        
        return convert_item(data)
