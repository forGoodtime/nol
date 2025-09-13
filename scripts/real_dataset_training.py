#!/usr/bin/env python3
"""
Real Car Damage Dataset Training Script
======================================

This script trains YOLOv8 models on the real car damage datasets available in the project.
It supports multiple multiclass classification datasets and can combine them for unified training.

Datasets supported:
- Car Scratch and Dent.v5i.multiclass (625 images): dent, dirt, scratch
- Rust and Scrach.v1i.multiclass (103 images): car, dent, rust, scratch  
- car scratch.v2i.multiclass (1221 images): clean, scratch, car-scratch

Usage:
    python scripts/real_dataset_training.py --dataset all --model yolov8n-cls.pt
    python scripts/real_dataset_training.py --dataset scratch_dent --epochs 100
    python scripts/real_dataset_training.py --dataset combined --batch-size 16
"""

import argparse
import os
import pandas as pd
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from datetime import datetime
import json

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATHS = {
    'scratch_dent': PROJECT_ROOT / 'Car Scratch and Dent.v5i.multiclass',
    'rust_scratch': PROJECT_ROOT / 'Rust and Scrach.v1i.multiclass', 
    'car_scratch': PROJECT_ROOT / 'car  scratch.v2i.multiclass'
}

def setup_directories():
    """Create necessary directories for training."""
    dirs = [
        PROJECT_ROOT / 'real_datasets',
        PROJECT_ROOT / 'real_datasets' / 'unified',
        PROJECT_ROOT / 'real_datasets' / 'unified' / 'train',
        PROJECT_ROOT / 'real_datasets' / 'unified' / 'val',
        PROJECT_ROOT / 'real_datasets' / 'unified' / 'test',
        PROJECT_ROOT / 'results' / 'real_training'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return PROJECT_ROOT / 'real_datasets' / 'unified'

def analyze_dataset(dataset_path):
    """Analyze a dataset and return class information."""
    print(f"\n📊 Analyzing dataset: {dataset_path.name}")
    
    splits = ['train', 'val', 'test']
    dataset_info = {
        'name': dataset_path.name,
        'path': dataset_path,
        'splits': {},
        'classes': [],
        'total_images': 0
    }
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            # Count images
            image_files = list(split_path.glob('*.jpg')) + list(split_path.glob('*.jpeg')) + list(split_path.glob('*.png'))
            
            # Read classes from CSV if exists
            classes_file = split_path / '_classes.csv'
            if classes_file.exists():
                df = pd.read_csv(classes_file)
                class_columns = [col for col in df.columns if col != 'filename']
                if not dataset_info['classes']:  # Set classes from first split found
                    dataset_info['classes'] = class_columns
                
                # Count class distributions
                class_counts = {}
                for class_name in class_columns:
                    class_counts[class_name] = df[class_name].sum()
                
                dataset_info['splits'][split] = {
                    'image_count': len(image_files),
                    'class_distribution': class_counts
                }
            else:
                dataset_info['splits'][split] = {
                    'image_count': len(image_files),
                    'class_distribution': {}
                }
            
            dataset_info['total_images'] += len(image_files)
    
    # Print analysis
    print(f"  📁 Total images: {dataset_info['total_images']}")
    print(f"  🏷️  Classes: {dataset_info['classes']}")
    
    for split, info in dataset_info['splits'].items():
        if info['image_count'] > 0:
            print(f"  📂 {split}: {info['image_count']} images")
            if info['class_distribution']:
                for class_name, count in info['class_distribution'].items():
                    print(f"     {class_name}: {count}")
    
    return dataset_info

def create_unified_classes(datasets_info):
    """Create unified class mapping from all datasets."""
    print("\n🔗 Creating unified class mapping...")
    
    # Collect all unique classes and normalize names
    all_classes = set()
    class_mapping = {}
    
    for dataset_info in datasets_info:
        for class_name in dataset_info['classes']:
            # Normalize class names
            normalized = class_name.lower().strip()
            
            # Map variations to standard names
            if normalized in ['dent', 'dunt']:
                standard = 'dent'
            elif normalized in ['scratch', 'scracth', 'car-scratch']:
                standard = 'scratch'
            elif normalized in ['rust']:
                standard = 'rust'
            elif normalized in ['dirt']:
                standard = 'dirt'
            elif normalized in ['car']:
                standard = 'car'
            elif normalized in ['0']:
                standard = 'clean'
            else:
                standard = normalized
            
            all_classes.add(standard)
            class_mapping[class_name] = standard
    
    unified_classes = sorted(list(all_classes))
    
    print(f"  📋 Unified classes: {unified_classes}")
    print(f"  🔄 Class mapping: {class_mapping}")
    
    return unified_classes, class_mapping

def prepare_unified_dataset(datasets_info, unified_classes, class_mapping, output_dir):
    """Prepare unified dataset in YOLOv8 classification format."""
    print("\n📦 Preparing unified dataset...")
    
    # Create class directories
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for class_name in unified_classes:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
    
    total_processed = 0
    class_counts = {split: {cls: 0 for cls in unified_classes} for split in ['train', 'val', 'test']}
    
    for dataset_info in datasets_info:
        print(f"  Processing {dataset_info['name']}...")
        
        for split in ['train', 'val', 'test']:
            if split not in dataset_info['splits']:
                continue
                
            split_path = dataset_info['path'] / split
            classes_file = split_path / '_classes.csv'
            
            if not classes_file.exists():
                print(f"    ⚠️ No classes file found for {split}")
                continue
            
            df = pd.read_csv(classes_file)
            
            for _, row in df.iterrows():
                filename = row['filename']
                image_path = split_path / filename
                
                if not image_path.exists():
                    continue
                
                # Determine the primary class for this image
                primary_class = None
                max_confidence = 0
                
                for original_class in dataset_info['classes']:
                    if original_class in row and row[original_class] > max_confidence:
                        max_confidence = row[original_class]
                        primary_class = class_mapping.get(original_class, original_class.lower())
                
                # If no positive class found, assign to 'clean' if available
                if primary_class is None or max_confidence == 0:
                    primary_class = 'clean' if 'clean' in unified_classes else unified_classes[0]
                
                # Copy image to appropriate class directory
                dest_dir = output_dir / split / primary_class
                
                # Create shorter filename to avoid filesystem limits
                base_name = Path(filename).stem
                extension = Path(filename).suffix
                short_dataset_name = dataset_info['name'][:20].replace(' ', '_').replace('.', '_')
                short_filename = f"{short_dataset_name}_{total_processed:06d}{extension}"
                
                dest_path = dest_dir / short_filename
                
                shutil.copy2(image_path, dest_path)
                class_counts[split][primary_class] += 1
                total_processed += 1
    
    print(f"  ✅ Processed {total_processed} images")
    
    # Print distribution
    for split in ['train', 'val', 'test']:
        split_total = sum(class_counts[split].values())
        if split_total > 0:
            print(f"  📊 {split} distribution:")
            for class_name, count in class_counts[split].items():
                if count > 0:
                    percentage = (count / split_total) * 100
                    print(f"     {class_name}: {count} ({percentage:.1f}%)")
    
    return class_counts

def create_dataset_yaml(output_dir, unified_classes):
    """Create dataset.yaml file for YOLOv8 classification training."""
    dataset_yaml = {
        'path': str(output_dir),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(unified_classes),
        'names': {i: name for i, name in enumerate(unified_classes)}
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"  📄 Created dataset.yaml: {yaml_path}")
    return yaml_path

def train_model(dataset_path, args):
    """Train YOLOv8 classification model."""
    print(f"\n🚀 Starting training with {args.model}...")
    
    # Initialize model
    model = YOLO(args.model)
    
    # Setup training parameters
    training_args = {
        'data': str(dataset_path),  # Use dataset directory
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'device': args.device,
        'patience': args.patience,
        'save_period': args.save_period,
        'project': str(PROJECT_ROOT / 'results' / 'real_training'),
        'name': f'real_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'verbose': True,
        'plots': True
    }
    
    # Add augmentation parameters
    if args.augment:
        training_args.update({
            'degrees': 15.0,        # rotation
            'translate': 0.1,       # translation
            'scale': 0.1,          # scaling
            'fliplr': 0.5,         # horizontal flip
            'flipud': 0.0,         # vertical flip
            'hsv_h': 0.015,        # hue augmentation
            'hsv_s': 0.7,          # saturation
            'hsv_v': 0.4,          # value
        })
    
    print(f"  📋 Training parameters:")
    for key, value in training_args.items():
        print(f"     {key}: {value}")
    
    # Start training
    results = model.train(**training_args)
    
    # Save training summary
    summary = {
        'model': args.model,
        'dataset': 'real_datasets_unified',
        'training_args': training_args,
        'results': str(results),
        'timestamp': datetime.now().isoformat()
    }
    
    results_dir = Path(training_args['project']) / training_args['name']
    with open(results_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results saved to: {results_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on real car damage datasets')
    
    # Dataset selection
    parser.add_argument('--dataset', choices=['all', 'scratch_dent', 'rust_scratch', 'car_scratch', 'combined'],
                       default='all', help='Dataset to use for training')
    
    # Model parameters
    parser.add_argument('--model', default='yolov8n-cls.pt',
                       help='YOLOv8 classification model (yolov8n-cls.pt, yolov8s-cls.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--device', default='cpu', help='Device to use (auto, cpu, 0, 1, etc.)')
    
    # Training options
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Analysis only
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze datasets without training')
    
    args = parser.parse_args()
    
    print("🚗 Real Car Damage Dataset Training")
    print("=" * 50)
    
    # Setup directories
    output_dir = setup_directories()
    
    # Select datasets to process
    if args.dataset == 'all':
        selected_datasets = list(DATASET_PATHS.keys())
    elif args.dataset == 'combined':
        selected_datasets = list(DATASET_PATHS.keys())
    else:
        selected_datasets = [args.dataset]
    
    # Analyze datasets
    datasets_info = []
    for dataset_name in selected_datasets:
        if dataset_name not in DATASET_PATHS:
            print(f"❌ Unknown dataset: {dataset_name}")
            continue
        
        dataset_path = DATASET_PATHS[dataset_name]
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            continue
        
        dataset_info = analyze_dataset(dataset_path)
        datasets_info.append(dataset_info)
    
    if not datasets_info:
        print("❌ No valid datasets found!")
        return
    
    if args.analyze_only:
        print("\n✅ Dataset analysis completed!")
        return
    
    # Create unified dataset
    unified_classes, class_mapping = create_unified_classes(datasets_info)
    class_counts = prepare_unified_dataset(datasets_info, unified_classes, class_mapping, output_dir)
    
    # Create dataset configuration
    dataset_yaml = create_dataset_yaml(output_dir, unified_classes)
    
    # Train model
    results = train_model(output_dir, args)
    
    print("\n🎉 Training pipeline completed successfully!")

if __name__ == '__main__':
    main()
