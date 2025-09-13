#!/usr/bin/env python3
"""
Скрипт для анализа результатов тестирования на реальных данных
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class RealResultsAnalyzer:
    """Анализатор результатов на реальных данных"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.load_results()
    
    def load_results(self):
        """Загрузка всех результатов"""
        print(f"📊 Загрузка результатов из {self.results_dir}")
        
        # Ищем все JSON файлы с результатами
        result_files = list(self.results_dir.rglob("*_result.json"))
        batch_files = list(self.results_dir.rglob("batch_results.json"))
        
        print(f"Найдено {len(result_files)} индивидуальных результатов")
        print(f"Найдено {len(batch_files)} пакетных результатов")
        
        # Загружаем индивидуальные результаты
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['source_file'] = str(result_file)
                    self.results_data.append(data)
            except Exception as e:
                print(f"❌ Ошибка загрузки {result_file}: {e}")
        
        # Загружаем пакетные результаты
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    for result in batch_data.get('results', []):
                        result['source_file'] = str(batch_file)
                        self.results_data.append(result)
            except Exception as e:
                print(f"❌ Ошибка загрузки {batch_file}: {e}")
        
        print(f"✅ Загружено {len(self.results_data)} результатов")
    
    def generate_summary_stats(self) -> Dict:
        """Генерация сводной статистики"""
        if not self.results_data:
            return {}
        
        # Извлекаем основные метрики
        damage_levels = [r['predictions']['damage_level'] for r in self.results_data]
        confidences = [r['predictions']['confidence'] for r in self.results_data]
        processing_times = [r['predictions']['processing_time'] for r in self.results_data]
        
        # Типы повреждений
        all_damage_types = []
        for r in self.results_data:
            all_damage_types.extend(r['predictions'].get('damage_types', []))
        
        # Размеры изображений
        image_sizes = []
        for r in self.results_data:
            size = r['metadata'].get('image_size', [0, 0])
            if len(size) >= 2:
                image_sizes.append(size[0] * size[1])  # площадь
        
        stats = {
            'total_images': len(self.results_data),
            'damage_level_distribution': dict(Counter(damage_levels)),
            'confidence_stats': {
                'mean': sum(confidences) / len(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'below_50': sum(1 for c in confidences if c < 0.5),
                'above_90': sum(1 for c in confidences if c > 0.9)
            },
            'processing_time_stats': {
                'mean': sum(processing_times) / len(processing_times) if processing_times else 0,
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0
            },
            'damage_types_frequency': dict(Counter(all_damage_types)),
            'image_size_stats': {
                'mean_area': sum(image_sizes) / len(image_sizes) if image_sizes else 0,
                'min_area': min(image_sizes) if image_sizes else 0,
                'max_area': max(image_sizes) if image_sizes else 0
            }
        }
        
        return stats
    
    def create_visualizations(self, output_dir: str):
        """Создание визуализаций"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.results_data:
            print("❌ Нет данных для визуализации")
            return
        
        # Подготовка данных для pandas
        df_data = []
        for r in self.results_data:
            df_data.append({
                'image_name': r['image_name'],
                'damage_level': r['predictions']['damage_level'],
                'confidence': r['predictions']['confidence'],
                'processing_time': r['predictions']['processing_time'],
                'num_damage_types': len(r['predictions'].get('damage_types', [])),
                'has_bbox': len(r['predictions'].get('bboxes', [])) > 0
            })
        
        df = pd.DataFrame(df_data)
        
        # 1. Распределение уровней повреждений
        plt.figure(figsize=(10, 6))
        damage_counts = df['damage_level'].value_counts().sort_index()
        
        plt.subplot(2, 2, 1)
        damage_counts.plot(kind='bar', color='skyblue')
        plt.title('Распределение уровней повреждений')
        plt.xlabel('Уровень повреждения')
        plt.ylabel('Количество изображений')
        plt.xticks([0, 1, 2, 3], ['Нет', 'Легкие', 'Средние', 'Серьезные'], rotation=45)
        
        # 2. Распределение confidence
        plt.subplot(2, 2, 2)
        plt.hist(df['confidence'], bins=20, alpha=0.7, color='lightgreen')
        plt.title('Распределение уверенности модели')
        plt.xlabel('Confidence')
        plt.ylabel('Количество')
        plt.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Среднее: {df["confidence"].mean():.3f}')
        plt.legend()
        
        # 3. Время обработки
        plt.subplot(2, 2, 3)
        plt.hist(df['processing_time'], bins=15, alpha=0.7, color='orange')
        plt.title('Время обработки')
        plt.xlabel('Время (сек)')
        plt.ylabel('Количество')
        plt.axvline(df['processing_time'].mean(), color='red', linestyle='--',
                   label=f'Среднее: {df["processing_time"].mean():.3f}с')
        plt.legend()
        
        # 4. Связь confidence и damage level
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='damage_level', y='confidence')
        plt.title('Confidence по уровням повреждений')
        plt.xlabel('Уровень повреждения')
        plt.ylabel('Confidence')
        plt.xticks([0, 1, 2, 3], ['Нет', 'Легкие', 'Средние', 'Серьезные'])
        
        plt.tight_layout()
        plt.savefig(output_path / 'analysis_overview.png', dpi=300, bbox_inches='tight')
        print(f"📊 Сохранена визуализация: {output_path / 'analysis_overview.png'}")
        plt.close()
        
        # Дополнительные графики если есть типы повреждений
        damage_types_data = []
        for r in self.results_data:
            for damage_type in r['predictions'].get('damage_types', []):
                damage_types_data.append({
                    'damage_type': damage_type,
                    'damage_level': r['predictions']['damage_level'],
                    'confidence': r['predictions']['confidence']
                })
        
        if damage_types_data:
            df_damage_types = pd.DataFrame(damage_types_data)
            
            plt.figure(figsize=(12, 8))
            
            # Частота типов повреждений
            plt.subplot(2, 2, 1)
            damage_type_counts = df_damage_types['damage_type'].value_counts()
            damage_type_counts.plot(kind='bar', color='coral')
            plt.title('Частота типов повреждений')
            plt.xlabel('Тип повреждения')
            plt.ylabel('Количество')
            plt.xticks(rotation=45)
            
            # Связь типов повреждений и уровней
            plt.subplot(2, 2, 2)
            damage_crosstab = pd.crosstab(df_damage_types['damage_type'], 
                                        df_damage_types['damage_level'])
            sns.heatmap(damage_crosstab, annot=True, fmt='d', cmap='Blues')
            plt.title('Типы повреждений × Уровни')
            plt.xlabel('Уровень повреждения')
            plt.ylabel('Тип повреждения')
            
            plt.tight_layout()
            plt.savefig(output_path / 'damage_types_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 Сохранена визуализация типов повреждений: {output_path / 'damage_types_analysis.png'}")
            plt.close()
    
    def generate_report(self, output_file: str):
        """Генерация текстового отчета"""
        stats = self.generate_summary_stats()
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО ТЕСТИРОВАНИЮ НА РЕАЛЬНЫХ ДАННЫХ")
        report.append("=" * 80)
        report.append(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Всего изображений: {stats['total_images']}")
        report.append("")
        
        # Распределение уровней повреждений
        report.append("РАСПРЕДЕЛЕНИЕ УРОВНЕЙ ПОВРЕЖДЕНИЙ:")
        damage_labels = {0: "Нет повреждений", 1: "Легкие", 2: "Средние", 3: "Серьезные"}
        for level, count in sorted(stats['damage_level_distribution'].items()):
            percentage = count / stats['total_images'] * 100
            report.append(f"  {damage_labels.get(level, f'Уровень {level}')}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Статистика уверенности
        conf_stats = stats['confidence_stats']
        report.append("СТАТИСТИКА УВЕРЕННОСТИ МОДЕЛИ:")
        report.append(f"  Средняя уверенность: {conf_stats['mean']:.3f}")
        report.append(f"  Минимальная: {conf_stats['min']:.3f}")
        report.append(f"  Максимальная: {conf_stats['max']:.3f}")
        report.append(f"  Низкая уверенность (<0.5): {conf_stats['below_50']} изображений")
        report.append(f"  Высокая уверенность (>0.9): {conf_stats['above_90']} изображений")
        report.append("")
        
        # Производительность
        perf_stats = stats['processing_time_stats']
        report.append("ПРОИЗВОДИТЕЛЬНОСТЬ:")
        report.append(f"  Среднее время обработки: {perf_stats['mean']:.3f} сек")
        report.append(f"  Минимальное время: {perf_stats['min']:.3f} сек")
        report.append(f"  Максимальное время: {perf_stats['max']:.3f} сек")
        report.append("")
        
        # Типы повреждений
        if stats['damage_types_frequency']:
            report.append("ЧАСТОТА ТИПОВ ПОВРЕЖДЕНИЙ:")
            for damage_type, count in sorted(stats['damage_types_frequency'].items(), 
                                           key=lambda x: x[1], reverse=True):
                report.append(f"  {damage_type}: {count}")
            report.append("")
        
        # Рекомендации
        report.append("РЕКОМЕНДАЦИИ:")
        
        if conf_stats['below_50'] > stats['total_images'] * 0.2:
            report.append("  ⚠️  Высокий процент предсказаний с низкой уверенностью")
            report.append("      → Рекомендуется доообучение модели")
        
        if perf_stats['mean'] > 1.0:
            report.append("  ⚠️  Медленная обработка изображений")
            report.append("      → Рассмотрите оптимизацию или использование более быстрой модели")
        
        no_damage_ratio = stats['damage_level_distribution'].get(0, 0) / stats['total_images']
        if no_damage_ratio > 0.8:
            report.append("  ℹ️  Большинство изображений без повреждений")
            report.append("      → Убедитесь в репрезентативности тестовой выборки")
        
        report.append("")
        report.append("=" * 80)
        
        # Сохраняем отчет
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📝 Отчет сохранен: {output_file}")
        
        # Выводим краткую версию в консоль
        print("\n" + "\n".join(report[:25]) + "\n...")


def main():
    parser = argparse.ArgumentParser(description="Анализ результатов тестирования")
    parser.add_argument("--results-dir", type=str, 
                       default="data/real_test_cases/results",
                       help="Папка с результатами")
    parser.add_argument("--output-dir", type=str,
                       default="data/real_test_cases/analysis",
                       help="Папка для аналитических отчетов")
    
    args = parser.parse_args()
    
    # Базовый путь к проекту
    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    
    if not results_dir.exists():
        print(f"❌ Папка с результатами не найдена: {results_dir}")
        print(f"💡 Сначала запустите тестирование:")
        print(f"   python scripts/test_real_cases.py")
        return
    
    # Создаем анализатор
    analyzer = RealResultsAnalyzer(str(results_dir))
    
    if not analyzer.results_data:
        print("❌ Не найдено данных для анализа")
        return
    
    # Создаем папку для результатов
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Генерация аналитического отчета...")
    
    # Генерируем визуализации
    analyzer.create_visualizations(str(output_dir))
    
    # Генерируем текстовый отчет
    report_file = output_dir / "analysis_report.txt"
    analyzer.generate_report(str(report_file))
    
    # Сохраняем детальную статистику в JSON
    stats = analyzer.generate_summary_stats()
    stats_file = output_dir / "detailed_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Анализ завершен! Результаты в папке: {output_dir}")
    print(f"📊 Графики: analysis_overview.png, damage_types_analysis.png")
    print(f"📝 Отчет: analysis_report.txt")
    print(f"📋 Статистика: detailed_stats.json")


if __name__ == "__main__":
    main()
