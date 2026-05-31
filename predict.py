"""
Standalone Predictor CLI untuk Coffee Defect Classification
Digunakan untuk testing model via Command Line/Terminal.
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import cv2
from utils.preprocessing import FeatureExtractor

def predict_single_image(image_path, model_path='models/coffee_defect_model.pkl', metadata_path='models/model_metadata.json'):
    # Check files
    if not os.path.exists(image_path):
        print(f"❌ Error: Image path '{image_path}' not found!")
        return None
        
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("❌ Error: Trained model or metadata files not found! Please run 'python train_model.py' first.")
        return None
        
    print(f"🔄 Loading model and metadata...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"🔄 Preprocessing & extracting features from {os.path.basename(image_path)}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: OpenCV cannot load image at '{image_path}'")
        return None
        
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features_from_image(img)
    features = features.reshape(1, -1)
    
    print(f"🤖 Predicting with Random Forest...")
    class_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    predicted_class = metadata['class_names'][class_idx]
    confidence = probabilities[class_idx] * 100
    
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULTS")
    print("="*50)
    print(f"Class Terprediksi  : {predicted_class}")
    print(f"Confidence Score   : {confidence:.2f}%")
    print("="*50)
    
    print("\n📊 Top 3 Candidates:")
    top_3_idx = np.argsort(probabilities)[::-1][:3]
    for rank, idx in enumerate(top_3_idx, 1):
        name = metadata['class_names'][idx]
        prob = probabilities[idx] * 100
        print(f" {rank}. {name:<25}: {prob:.2f}%")
        
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coffee Defect Standalone Predictor CLI")
    parser.add_argument('--image', type=str, required=True, help='Path to target coffee bean image')
    parser.add_argument('--model', type=str, default='models/coffee_defect_model.pkl', help='Path to model pickle file')
    parser.add_argument('--metadata', type=str, default='models/model_metadata.json', help='Path to metadata JSON file')
    
    args = parser.parse_args()
    predict_single_image(args.image, args.model, args.metadata)
