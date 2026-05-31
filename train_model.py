"""
Training Script untuk Random Forest Model
Klasifikasi Cacat Biji Kopi dengan 17 Kelas
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import FeatureExtractor
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'dataset_path': './archive',  # Path ke folder dataset (gunakan archive yang berisi folder kelas)
    'model_output_path': './models',
    'test_size': 0.2,
    'random_state': 42,
    'random_forest_params': {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
}

class CoffeeDefectModelTrainer:
    """Trainer untuk Coffee Defect Classification Model"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_history = {}
        
        # Buat output folder jika belum ada
        os.makedirs(config['model_output_path'], exist_ok=True)
    
    def load_dataset(self):
        """Load dataset dari folder terstruktur"""
        print("[INFO] Loading dataset...")
        
        dataset_path = self.config['dataset_path']
        X = []
        y = []
        class_names = []
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path tidak ditemukan: {dataset_path}")
        
        # Iterasi setiap kelas (folder)
        for class_idx, class_name in enumerate(sorted(os.listdir(dataset_path))):
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            class_names.append(class_name)
            print(f"  [CLASS] Processing class: {class_name}")
            
            # Load semua gambar dalam folder
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"    Found {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Ekstrak fitur
                    features = self.feature_extractor.extract_features_from_image(img_path)
                    X.append(features)
                    y.append(class_idx)
                except Exception as e:
                    print(f"    [WARN] Error loading {img_file}: {str(e)}")
                    continue
        
        X = np.array(X)
        y = np.array(y)
        self.class_names = np.array(class_names)
        
        print(f"\n[SUCCESS] Dataset loaded:")
        print(f"   Total samples: {len(X)}")
        print(f"   Feature dimension: {X.shape[1]}")
        print(f"   Number of classes: {len(class_names)}")
        print(f"   Classes: {', '.join(class_names)}")
        
        return X, y
    
    def train(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Load dataset
        X, y = self.load_dataset()
        
        # Split data
        print("\n[INFO] Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train model
        print("\n[INFO] Training RandomForestClassifier...")
        print(f"   Parameters: {self.config['random_forest_params']}")
        
        self.model = RandomForestClassifier(**self.config['random_forest_params'])
        self.model.fit(X_train, y_train)
        
        print("   [SUCCESS] Training complete!")
        
        # Evaluate model
        print("\n[INFO] Evaluating model...")
        self.evaluate()
        
        return self.model
    
    def evaluate(self):
        """Evaluate model performance"""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Training metrics
        train_acc = accuracy_score(self.y_train, y_pred_train)
        
        # Testing metrics
        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_precision = precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        
        # Classification report
        class_report = classification_report(
            self.y_test, y_pred_test,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Store metrics
        self.training_history = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'n_features': self.X_train.shape[1],
            'n_classes': len(self.class_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        # Print metrics
        print(f"\n   Training Accuracy:  {train_acc:.4f}")
        print(f"   Testing Accuracy:   {test_acc:.4f}")
        print(f"   Precision (weighted): {test_precision:.4f}")
        print(f"   Recall (weighted):    {test_recall:.4f}")
        print(f"   F1-Score (weighted):  {test_f1:.4f}")
        
        print("\n[REPORT] Classification Report:")
        print(classification_report(
            self.y_test, y_pred_test,
            target_names=self.class_names,
            zero_division=0
        ))
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confusion_matrix': cm,
            'y_pred': y_pred_test
        }
    
    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.config['model_output_path'], 'coffee_defect_model.pkl')
        
        print(f"\n[SAVE] Saving model to {model_path}...")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"   [SUCCESS] Model saved!")
        
        return model_path
    
    def save_model_metadata(self):
        """Save model metadata (class names, feature info, metrics)"""
        metadata_path = os.path.join(
            self.config['model_output_path'],
            'model_metadata.json'
        )
        
        metadata = {
            'class_names': self.class_names.tolist(),
            'training_history': self.training_history,
            'model_params': self.config['random_forest_params'],
            'feature_extractor_info': {
                'total_features': self.training_history['n_features'],
                'feature_types': [
                    'HOG (324)',
                    'LBP Texture (59)',
                    'Color Histogram (32)',
                    'Edge Features (33)',
                    'Morphological (3)'
                ]
            }
        }
        
        print(f"[SAVE] Saving metadata to {metadata_path}...")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"   [SUCCESS] Metadata saved!")
    
    def save_visualizations(self):
        """Save training visualizations"""
        print("\n[INFO] Creating visualizations...")
        
        viz_dir = os.path.join(self.config['model_output_path'], 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(16, 12))
        cm = np.array(self.training_history['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Coffee Defect Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        print("   [SUCCESS] Confusion matrix saved")
        plt.close()
        
        # 2. Feature Importance
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 20 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        print("   [SUCCESS] Feature importance saved")
        plt.close()
        
        # 3. Class Distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        plt.bar(self.class_names[unique], counts)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Training Data Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
        print("   [SUCCESS] Class distribution saved")
        plt.close()
        
        print(f"   [INFO] Visualizations saved to {viz_dir}")
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        try:
            # Train
            self.train()
            
            # Save model
            self.save_model()
            
            # Save metadata
            self.save_model_metadata()
            
            # Save visualizations
            self.save_visualizations()
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)
            print(f"Model saved to: ./models/coffee_defect_model.pkl")
            print(f"Metadata saved to: ./models/model_metadata.json")
            print(f"Visualizations saved to: ./models/visualizations/")
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    # Create trainer
    trainer = CoffeeDefectModelTrainer(CONFIG)
    
    # Run training
    trainer.run_full_pipeline()
