#!/usr/bin/env python3
"""
Скрипт для сравнения результатов инференса с Ground Truth данными
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class GTComparator:
    """Сравнение с Ground Truth данными"""
    
    def __init__(self, results_dir: str, gt_dir: str):
        self.results_dir = Path(results_dir)
        self.gt_dir = Path(gt_dir)
        
        self.predictions = {}
        self.ground_truth = {}
        
        self.damage_labels = {
            0: "no_damage",
            1: "light_damage",
            2: "moderate_damage", 
            3: "severe_damage"
        }
    
    def load_predictions(self):
        """Загрузка предсказаний модели"""
        print("📊 Загрузка предсказаний...")
        
        for result_file in self.results_dir.rglob("*_result.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    image_name = data['image_name']
                    self.predictions[image_name] = {
                        'damage_level': data['predictions']['damage_level'],
                        'confidence': data['predictions']['confidence'],
                        'damage_types': data['predictions'].get('damage_types', [])
                    }
            except Exception as e:
                print(f"❌ Ошибка загрузки {result_file}: {e}")
        
        print(f"✅ Загружено {len(self.predictions)} предсказаний")
    
    def load_ground_truth(self):
        """Загрузка Ground Truth данных"""
        print("🎯 Загрузка Ground Truth...")
        
        # Ищем файлы с аннотациями
        gt_files = list(self.gt_dir.rglob("*.json"))
        
        if not gt_files:
            print(f"❌ Не найдено GT файлов в {self.gt_dir}")
            return False
        
        for gt_file in gt_files:
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Поддержка разных форматов GT
                    if 'images' in data and 'annotations' in data:
                        # COCO формат
                        self._load_coco_gt(data)
                    elif 'ground_truth' in data:
                        # Кастомный формат
                        self._load_custom_gt(data)
                    else:
                        # Простой формат: {image_name: {damage_level: X, ...}}
                        for image_name, gt_data in data.items():
                            if isinstance(gt_data, dict) and 'damage_level' in gt_data:
                                self.ground_truth[image_name] = gt_data
                                
            except Exception as e:
                print(f"❌ Ошибка загрузки GT {gt_file}: {e}")
        
        print(f"✅ Загружено {len(self.ground_truth)} GT аннотаций")
        return len(self.ground_truth) > 0
    
    def _load_coco_gt(self, coco_data):
        """Загрузка COCO формата"""
        # Создаем маппинг image_id -> filename
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Группируем аннотации по изображениям
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Преобразуем в наш формат
        for image_id, annotations in image_annotations.items():
            image_name = image_map.get(image_id, f"image_{image_id}")
            
            # Определяем максимальный уровень повреждения
            max_damage_level = 0
            damage_types = set()
            
            for ann in annotations:
                category_id = ann.get('category_id', 0)
                # Маппинг category_id в damage_level (настраивается)
                damage_level = min(category_id, 3)  # Ограничиваем 3
                max_damage_level = max(max_damage_level, damage_level)
                
                # Извлекаем тип повреждения из attributes
                if 'attributes' in ann:
                    damage_types.update(ann['attributes'].get('damage_types', []))
            
            self.ground_truth[image_name] = {
                'damage_level': max_damage_level,
                'damage_types': list(damage_types)
            }
    
    def _load_custom_gt(self, data):
        """Загрузка кастомного формата"""
        for gt_item in data['ground_truth']:
            image_name = gt_item['image_name']
            self.ground_truth[image_name] = {
                'damage_level': gt_item['damage_level'],
                'damage_types': gt_item.get('damage_types', [])
            }
    
    def compare_predictions(self) -> Dict:
        """Сравнение предсказаний с GT"""
        if not self.predictions or not self.ground_truth:
            print("❌ Нет данных для сравнения")
            return {}
        
        # Находим общие изображения
        common_images = set(self.predictions.keys()) & set(self.ground_truth.keys())
        
        if not common_images:
            print("❌ Нет общих изображений между предсказаниями и GT")
            print(f"Предсказания: {list(self.predictions.keys())[:5]}...")
            print(f"GT: {list(self.ground_truth.keys())[:5]}...")
            return {}
        
        print(f"🔍 Найдено {len(common_images)} общих изображений")
        
        # Собираем данные для сравнения
        y_true = []
        y_pred = []
        confidences = []
        
        detailed_results = []
        
        for image_name in common_images:
            gt_level = self.ground_truth[image_name]['damage_level']
            pred_level = self.predictions[image_name]['damage_level']
            confidence = self.predictions[image_name]['confidence']
            
            y_true.append(gt_level)
            y_pred.append(pred_level)
            confidences.append(confidence)
            
            detailed_results.append({
                'image_name': image_name,
                'gt_level': gt_level,
                'pred_level': pred_level,
                'confidence': confidence,
                'correct': gt_level == pred_level
            })
        
        # Вычисляем метрики
        accuracy = accuracy_score(y_true, y_pred)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        
        # Классификационный отчет
        class_report = classification_report(
            y_true, y_pred, 
            target_names=[self.damage_labels[i] for i in range(4)],
            output_dict=True
        )
        
        # Анализ по уровням уверенности
        confidence_analysis = self._analyze_by_confidence(
            detailed_results, confidences
        )
        
        results = {
            'total_images': len(common_images),
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'confidence_analysis': confidence_analysis,
            'detailed_results': detailed_results
        }
        
        return results
    
    def _analyze_by_confidence(self, detailed_results: List[Dict], 
                             confidences: List[float]) -> Dict:
        """Анализ точности по уровням уверенности"""
        confidence_thresholds = [0.5, 0.7, 0.8, 0.9]
        analysis = {}
        
        for threshold in confidence_thresholds:
            high_conf_results = [r for r, c in zip(detailed_results, confidences) 
                               if c >= threshold]
            
            if high_conf_results:
                correct = sum(1 for r in high_conf_results if r['correct'])
                accuracy = correct / len(high_conf_results)
                
                analysis[f'confidence_{threshold}'] = {
                    'count': len(high_conf_results),
                    'accuracy': accuracy,
                    'percentage_of_total': len(high_conf_results) / len(detailed_results) * 100
                }
        
        return analysis
    
    def create_visualizations(self, results: Dict, output_dir: str):
        """Создание визуализаций сравнения"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Матрица ошибок
        plt.figure(figsize=(10, 8))
        
        cm = np.array(results['confusion_matrix'])
        
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.damage_labels[i] for i in range(4)],
                    yticklabels=[self.damage_labels[i] for i in range(4)])
        plt.title(f'Матрица ошибок\nТочность: {results["accuracy"]:.3f}')
        plt.xlabel('Предсказание')
        plt.ylabel('Ground Truth')
        
        # 2. Точность по уровням уверенности
        plt.subplot(2, 2, 2)
        conf_analysis = results['confidence_analysis']
        thresholds = []
        accuracies = []
        counts = []
        
        for key, data in conf_analysis.items():
            threshold = float(key.split('_')[1])
            thresholds.append(threshold)
            accuracies.append(data['accuracy'])
            counts.append(data['count'])
        
        plt.plot(thresholds, accuracies, 'bo-', label='Точность')
        plt.xlabel('Порог уверенности')
        plt.ylabel('Точность')
        plt.title('Точность vs Уверенность')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Распределение ошибок
        plt.subplot(2, 2, 3)
        detailed = results['detailed_results']
        correct_conf = [r['confidence'] for r in detailed if r['correct']]
        incorrect_conf = [r['confidence'] for r in detailed if not r['correct']]
        
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Ошибки', color='red')
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Правильные', color='green')
        plt.xlabel('Уверенность')
        plt.ylabel('Количество')
        plt.title('Распределение уверенности')
        plt.legend()
        
        # 4. Диаграмма по классам
        plt.subplot(2, 2, 4)
        class_report = results['classification_report']
        classes = [self.damage_labels[i] for i in range(4)]
        f1_scores = [class_report[cls]['f1-score'] for cls in classes 
                    if cls in class_report]
        
        plt.bar(range(len(f1_scores)), f1_scores, color='skyblue')
        plt.xlabel('Класс повреждения')
        plt.ylabel('F1-Score')
        plt.title('F1-Score по классам')
        plt.xticks(range(len(f1_scores)), classes, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'gt_comparison.png', dpi=300, bbox_inches='tight')
        print(f"📊 Визуализация сохранена: {output_path / 'gt_comparison.png'}")
        plt.close()
    
    def generate_report(self, results: Dict, output_file: str):
        """Генерация отчета сравнения"""
        if not results:
            return
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ СРАВНЕНИЯ С GROUND TRUTH")
        report.append("=" * 80)
        report.append(f"Всего изображений для сравнения: {results['total_images']}")
        report.append(f"Общая точность: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        report.append("")
        
        # Классификационный отчет
        report.append("МЕТРИКИ ПО КЛАССАМ:")
        class_report = results['classification_report']
        for class_name in [self.damage_labels[i] for i in range(4)]:
            if class_name in class_report:
                metrics = class_report[class_name]
                report.append(f"  {class_name}:")
                report.append(f"    Precision: {metrics['precision']:.3f}")
                report.append(f"    Recall: {metrics['recall']:.3f}")
                report.append(f"    F1-Score: {metrics['f1-score']:.3f}")
                report.append(f"    Support: {int(metrics['support'])}")
        report.append("")
        
        # Анализ по уверенности
        report.append("АНАЛИЗ ПО УРОВНЯМ УВЕРЕННОСТИ:")
        conf_analysis = results['confidence_analysis']
        for key in sorted(conf_analysis.keys()):
            threshold = key.split('_')[1]
            data = conf_analysis[key]
            report.append(f"  Уверенность >= {threshold}:")
            report.append(f"    Количество: {data['count']} ({data['percentage_of_total']:.1f}%)")
            report.append(f"    Точность: {data['accuracy']:.3f}")
        report.append("")
        
        # Анализ ошибок
        detailed = results['detailed_results']
        errors = [r for r in detailed if not r['correct']]
        
        if errors:
            report.append("АНАЛИЗ ОШИБОК:")
            report.append(f"  Всего ошибок: {len(errors)}")
            
            # Группируем ошибки по типам
            error_patterns = {}
            for error in errors:
                pattern = f"GT:{error['gt_level']} → Pred:{error['pred_level']}"
                if pattern not in error_patterns:
                    error_patterns[pattern] = []
                error_patterns[pattern].append(error)
            
            report.append("  Частые ошибки:")
            for pattern, pattern_errors in sorted(error_patterns.items(), 
                                                key=lambda x: len(x[1]), reverse=True):
                avg_conf = sum(e['confidence'] for e in pattern_errors) / len(pattern_errors)
                report.append(f"    {pattern}: {len(pattern_errors)} случаев "
                             f"(сред. уверенность: {avg_conf:.3f})")
        
        report.append("")
        report.append("=" * 80)
        
        # Сохраняем отчет
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📝 Отчет сравнения сохранен: {output_file}")
        
        # Выводим в консоль
        print("\n" + "\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Сравнение с Ground Truth")
    parser.add_argument("--results-dir", type=str,
                       default="data/real_test_cases/results",
                       help="Папка с результатами инференса")
    parser.add_argument("--gt-dir", type=str,
                       default="data/annotations",
                       help="Папка с Ground Truth аннотациями")
    parser.add_argument("--output-dir", type=str,
                       default="data/real_test_cases/gt_comparison",
                       help="Папка для результатов сравнения")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    gt_dir = project_root / args.gt_dir
    output_dir = project_root / args.output_dir
    
    if not results_dir.exists():
        print(f"❌ Папка с результатами не найдена: {results_dir}")
        return
    
    if not gt_dir.exists():
        print(f"❌ Папка с GT не найдена: {gt_dir}")
        return
    
    # Создаем компаратор
    comparator = GTComparator(str(results_dir), str(gt_dir))
    
    # Загружаем данные
    comparator.load_predictions()
    if not comparator.load_ground_truth():
        print("❌ Не удалось загрузить Ground Truth данные")
        return
    
    # Сравниваем
    results = comparator.compare_predictions()
    
    if not results:
        print("❌ Сравнение не удалось")
        return
    
    # Создаем папку для результатов
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Генерируем визуализации
    comparator.create_visualizations(results, str(output_dir))
    
    # Генерируем отчет
    report_file = output_dir / "gt_comparison_report.txt"
    comparator.generate_report(results, str(report_file))
    
    # Сохраняем детальные результаты
    results_file = output_dir / "detailed_comparison.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Сравнение завершено! Результаты в: {output_dir}")


if __name__ == "__main__":
    main()
