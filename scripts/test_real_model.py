#!/usr/bin/env python3
"""
Test script for real car damage classification model.
Tests the trained YOLOv8 model on sample images.
"""

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class RealModelTester:
    """Test real car damage classification model."""
    
    def __init__(self, model_path):
        """Initialize with trained model."""
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.classes = ['car', 'clean', 'dent', 'dirt', 'rust', 'scratch']
        print(f"🚗 Loaded model: {model_path}")
        print(f"📋 Classes: {self.classes}")
    
    def predict_image(self, image_path, conf_threshold=0.3):
        """Predict single image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Predict
        results = self.model(str(image_path), conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Get prediction
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            class_id = np.argmax(probs)
            confidence = probs[class_id]
            predicted_class = self.classes[class_id]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {self.classes[i]: probs[i] for i in range(len(self.classes))},
                'image_path': str(image_path)
            }
        else:
            return None
    
    def test_directory(self, test_dir, output_dir=None):
        """Test all images in a directory."""
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.rglob(f'*{ext}'))
            image_files.extend(test_dir.rglob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in {test_dir}")
            return []
        
        print(f"🔍 Testing {len(image_files)} images from {test_dir}")
        
        results = []
        true_labels = []
        predicted_labels = []
        
        for img_path in image_files:
            try:
                # Get true label from parent directory if available
                true_label = img_path.parent.name
                if true_label in self.classes:
                    true_labels.append(true_label)
                else:
                    true_label = None
                
                # Predict
                prediction = self.predict_image(img_path)
                if prediction:
                    if true_label:
                        predicted_labels.append(prediction['predicted_class'])
                    
                    results.append({
                        **prediction,
                        'true_class': true_label,
                        'correct': true_label == prediction['predicted_class'] if true_label else None
                    })
                    
                    print(f"📸 {img_path.name}: {prediction['predicted_class']} ({prediction['confidence']:.3f})"
                          + (f" - {'✅' if prediction['correct'] else '❌'}" if true_label else ""))
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        
        # Calculate accuracy if we have true labels
        if true_labels and predicted_labels:
            accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels)
            print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Generate classification report
            print("\n📋 Classification Report:")
            print(classification_report(true_labels, predicted_labels, target_names=self.classes, zero_division=0))
            
            # Save confusion matrix if output directory specified
            if output_dir:
                self.save_confusion_matrix(true_labels, predicted_labels, output_dir)
        
        # Save detailed results
        if output_dir:
            self.save_results(results, output_dir)
        
        return results
    
    def save_confusion_matrix(self, true_labels, predicted_labels, output_dir):
        """Save confusion matrix visualization."""
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Car Damage Classification - Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        cm_path = output_dir / f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Confusion matrix saved: {cm_path}")
    
    def save_results(self, results, output_dir):
        """Save detailed results to file."""
        results_path = output_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(results_path, 'w') as f:
            f.write("🚗 Real Car Damage Model Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Classes: {self.classes}\n")
            f.write(f"Total images tested: {len(results)}\n\n")
            
            correct_predictions = sum(1 for r in results if r.get('correct') == True)
            total_with_labels = sum(1 for r in results if r.get('true_class') is not None)
            
            if total_with_labels > 0:
                accuracy = correct_predictions / total_with_labels
                f.write(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"Image: {Path(result['image_path']).name}\n")
                f.write(f"Predicted: {result['predicted_class']} ({result['confidence']:.3f})\n")
                if result.get('true_class'):
                    f.write(f"True class: {result['true_class']}\n")
                    f.write(f"Correct: {'Yes' if result.get('correct') else 'No'}\n")
                
                f.write("Probabilities:\n")
                for cls, prob in result['probabilities'].items():
                    f.write(f"  {cls}: {prob:.3f}\n")
                f.write("\n")
        
        print(f"💾 Detailed results saved: {results_path}")
    
    def visualize_predictions(self, image_paths, output_dir=None):
        """Visualize predictions on images."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        for img_path in image_paths[:9]:  # Show up to 9 images
            try:
                prediction = self.predict_image(img_path)
                if not prediction:
                    continue
                
                # Load and display image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.axis('off')
                
                title = f"{prediction['predicted_class']} ({prediction['confidence']:.3f})"
                plt.title(title, fontsize=14, fontweight='bold')
                
                if output_dir:
                    img_name = Path(img_path).stem
                    save_path = output_dir / f'prediction_{img_name}.png'
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    print(f"💾 Saved visualization: {save_path}")
                else:
                    plt.show()
                    
            except Exception as e:
                print(f"❌ Error visualizing {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test real car damage classification model')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--test-dir', help='Directory containing test images')
    parser.add_argument('--test-image', help='Single image to test')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Initialize tester
    tester = RealModelTester(args.model)
    
    if args.test_image:
        # Test single image
        print(f"\n🔍 Testing single image: {args.test_image}")
        prediction = tester.predict_image(args.test_image, args.conf)
        if prediction:
            print(f"📊 Prediction: {prediction['predicted_class']} ({prediction['confidence']:.3f})")
            print("📋 All probabilities:")
            for cls, prob in prediction['probabilities'].items():
                print(f"  {cls}: {prob:.3f}")
        
        if args.visualize:
            tester.visualize_predictions([args.test_image], args.output_dir)
    
    elif args.test_dir:
        # Test directory
        print(f"\n🔍 Testing directory: {args.test_dir}")
        results = tester.test_directory(args.test_dir, args.output_dir)
        
        if args.visualize and results:
            image_paths = [r['image_path'] for r in results[:9]]
            tester.visualize_predictions(image_paths, args.output_dir)
    
    else:
        print("❌ Please specify --test-image or --test-dir")


if __name__ == "__main__":
    main()
