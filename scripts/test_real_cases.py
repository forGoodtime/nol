#!/usr/bin/env python3
"""
Скрипт для тестирования модели на реальных данных без аннотаций
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Добавляем src в путь
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from utils.shadow_removal import ShadowRemover
except ImportError:
    print("⚠️ Shadow removal module not available, using basic preprocessing")
    ShadowRemover = None

try:
    from inference.infer import DamageInference
except ImportError:
    print("⚠️ Inference module not available, using mock inference")
    DamageInference = None

try:
    from cv_analysis.comparison_visualizer import ComparisonVisualizer
    CV_AVAILABLE = True
except ImportError:
    print("⚠️ CV analysis module not available")
    CV_AVAILABLE = False


class RealCasesTester:
    """Тестер для реальных случаев"""
    
    def __init__(self, model_path: Optional[str] = None, use_cv_analysis: bool = True):
        self.model_path = model_path
        self.use_cv_analysis = use_cv_analysis and CV_AVAILABLE
        
        # Инициализируем shadow remover если доступен
        if ShadowRemover:
            self.shadow_remover = ShadowRemover()
        else:
            self.shadow_remover = None
        
        # Инициализируем inference engine если доступен
        self.inference_engine = None
        if DamageInference and model_path and os.path.exists(model_path):
            self.inference_engine = DamageInference(model_path)
        
        # Инициализируем CV анализатор
        if self.use_cv_analysis:
            self.cv_comparator = ComparisonVisualizer()
            print("✅ CV анализ включен")
        else:
            self.cv_comparator = None
        
        self.damage_classes = {
            0: "no_damage",
            1: "light_damage", 
            2: "moderate_damage",
            3: "severe_damage"
        }
        
    def process_image(self, image_path: str) -> Dict:
        """Обработка одного изображения с CV и ML анализом"""
        start_time = time.time()
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        original_shape = image.shape[:2]
        
        # Предобработка изображения
        if self.shadow_remover:
            processed_image = self.shadow_remover.remove_shadows(image)
        else:
            # Базовая предобработка без shadow removal
            processed_image = cv2.GaussianBlur(image, (3, 3), 0)
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1.1, beta=10)
        
        # CV + ML анализ если доступен
        if self.use_cv_analysis and self.cv_comparator:
            try:
                combined_results = self.cv_comparator.analyze_with_both_methods(
                    processed_image, self.model_path
                )
                
                # Извлекаем ML результаты из комбинированного анализа
                ml_results = combined_results['ml_results']
                cv_results = combined_results['cv_results']
                
                # Сохраняем визуализации
                filename_base = Path(image_path).stem
                output_dir = Path(image_path).parent.parent / "results" / "cv_analysis" / filename_base
                
                saved_files = self.cv_comparator.save_comparison_results(
                    output_dir, filename_base, processed_image, combined_results
                )
                
            except Exception as e:
                print(f"   ⚠️ Ошибка CV анализа: {e}")
                # Fallback на mock inference
                ml_results = self._mock_inference(processed_image)
                cv_results = None
                saved_files = {}
        else:
            # Только ML инференс (или mock)
            if self.inference_engine:
                ml_results = self.inference_engine.predict(processed_image)
            else:
                ml_results = self._mock_inference(processed_image)
            
            cv_results = None
            saved_files = {}
        
        processing_time = time.time() - start_time
        
        # Формируем результат
        result = {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "predictions": {
                "damage_level": ml_results.get("damage_level", 0),
                "confidence": ml_results.get("confidence", 0.5),
                "damage_types": ml_results.get("damage_types", []),
                "bboxes": ml_results.get("bboxes", []),
                "processing_time": round(processing_time, 3)
            },
            "preprocessing": {
                "shadow_removed": self.shadow_remover is not None,
                "original_size": original_shape,
                "processed_size": processed_image.shape[:2]
            },
            "metadata": {
                "model_path": self.model_path or "mock_model",
                "timestamp": datetime.now().isoformat(),
                "image_size": original_shape,
                "cv_analysis_used": self.use_cv_analysis
            }
        }
        
        # Добавляем CV результаты если доступны
        if cv_results:
            result["cv_analysis"] = {
                "damage_regions": len(cv_results["damage_regions"]),
                "overall_damage_level": cv_results["overall_damage_level"],
                "damage_counts": cv_results["damage_counts"],
                "total_damaged_area": cv_results["total_damaged_area"]
            }
        
        # Добавляем пути к сохраненным файлам
        if saved_files:
            result["saved_files"] = saved_files
        
        return result
    
    def _mock_inference(self, image: np.ndarray) -> Dict:
        """Имитация инференса для демо"""
        # Простая эвристика основанная на характеристиках изображения
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Анализ текстуры и контрастности
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_intensity = np.mean(gray)
        
        # Простая логика для имитации
        if laplacian_var < 100:
            damage_level = 0
            confidence = 0.95
            damage_types = []
        elif laplacian_var < 500:
            damage_level = 1
            confidence = 0.78
            damage_types = ["light_scratches"]
        elif mean_intensity < 80:
            damage_level = 3
            confidence = 0.91
            damage_types = ["severe_damage", "missing_part"]
        else:
            damage_level = 2
            confidence = 0.83
            damage_types = ["rust", "dent"]
        
        # Фиктивные bounding boxes
        h, w = image.shape[:2]
        bboxes = []
        if damage_level > 0:
            bboxes.append({
                "bbox": [w//4, h//4, w//2, h//2],
                "confidence": confidence,
                "class": self.damage_classes[damage_level]
            })
        
        return {
            "damage_level": damage_level,
            "confidence": confidence,
            "damage_types": damage_types,
            "bboxes": bboxes
        }
    
    def test_directory(self, input_dir: str, output_dir: str) -> List[Dict]:
        """Тестирование всех изображений в папке"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поддерживаемые форматы
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Находим все изображения
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️ Не найдено изображений в {input_dir}")
            return []
        
        print(f"🔍 Найдено {len(image_files)} изображений")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"📸 Обрабатывается ({i}/{len(image_files)}): {image_file.name}")
            
            try:
                result = self.process_image(str(image_file))
                results.append(result)
                
                # Сохраняем индивидуальный результат
                result_file = output_path / f"{image_file.stem}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
                print(f"   ✅ Damage level: {result['predictions']['damage_level']} "
                      f"(confidence: {result['predictions']['confidence']:.3f})")
                
            except Exception as e:
                print(f"   ❌ Ошибка обработки {image_file.name}: {e}")
                continue
        
        # Сохраняем общий отчет
        summary_file = output_path / "batch_results.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_images": len(image_files),
                    "processed_successfully": len(results),
                    "failed": len(image_files) - len(results),
                    "processing_date": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Результаты сохранены в {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели на реальных данных с CV анализом")
    parser.add_argument("--input", choices=["production", "validation", "all"], 
                       default="production",
                       help="Какие данные тестировать")
    parser.add_argument("--model", type=str, default=None,
                       help="Путь к обученной модели")
    parser.add_argument("--custom-input", type=str, default=None,
                       help="Кастомный путь к папке с изображениями")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Папка для результатов")
    parser.add_argument("--no-cv-analysis", action="store_true",
                       help="Отключить CV анализ")
    
    args = parser.parse_args()
    
    # Базовый путь к проекту
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "real_test_cases"
    
    # Создаем тестер
    use_cv = not args.no_cv_analysis
    tester = RealCasesTester(args.model, use_cv_analysis=use_cv)
    
    if CV_AVAILABLE and use_cv:
        print("🔬 CV анализ включен - будут созданы детальные визуализации")
    else:
        print("📊 Используется только ML анализ")
    
    if args.custom_input:
        # Тестируем кастомную папку
        input_dir = args.custom_input
        output_dir = args.output_dir or str(data_root / "results" / "custom")
        
        print(f"🚗 Тестирование кастомной папки: {input_dir}")
        results = tester.test_directory(input_dir, output_dir)
        
    elif args.input == "all":
        # Тестируем все папки
        all_results = []
        
        for folder in ["production", "validation"]:
            input_dir = str(data_root / folder)
            output_dir = str(data_root / "results" / f"{folder}_results")
            
            if os.path.exists(input_dir):
                print(f"\n🚗 Тестирование {folder}...")
                results = tester.test_directory(input_dir, output_dir)
                all_results.extend(results)
            else:
                print(f"⚠️ Папка {input_dir} не существует")
        
        print(f"\n📊 Всего обработано изображений: {len(all_results)}")
        
    else:
        # Тестируем конкретную папку
        input_dir = str(data_root / args.input)
        output_dir = args.output_dir or str(data_root / "results" / f"{args.input}_results")
        
        if not os.path.exists(input_dir):
            print(f"❌ Папка {input_dir} не существует")
            print(f"💡 Создайте папку и добавьте изображения:")
            print(f"   mkdir -p {input_dir}")
            print(f"   cp /path/to/your/images/* {input_dir}/")
            return
        
        print(f"🚗 Тестирование {args.input}...")
        results = tester.test_directory(input_dir, output_dir)
    
    print("\n✅ Тестирование завершено!")
    
    if CV_AVAILABLE and use_cv:
        print("🔬 CV анализ результаты:")
        print("   • Сравнительные визуализации: *_cv_ml_comparison.png")
        print("   • Детальные отчеты: *_detailed_report.txt")
        print("   • Тепловые карты: *_heatmap.jpg")
        print("   • Маски повреждений: *_damage_mask.jpg")
    
    print(f"💡 Для общего анализа результатов используйте:")
    print(f"   python scripts/analyze_real_results.py")


if __name__ == "__main__":
    main()
