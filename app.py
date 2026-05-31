"""
Flask Backend untuk Coffee Defect Classification Web Application
Premium AI-powered Coffee Bean Defect Detection System
"""

import os
import json
import base64
import pickle
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from io import BytesIO
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Import preprocessing utilities
from utils.preprocessing import FeatureExtractor

# ========================
# Flask App Initialization
# ========================
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'coffee-defect-classifier-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Vercel: filesystem read-only kecuali /tmp
app.config['UPLOAD_FOLDER'] = '/tmp/uploads' if os.environ.get('VERCEL') else 'uploads'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

# ========================
# Global Variables
# ========================
model = None
model_metadata = None
feature_extractor = None
prediction_history = []
MAX_HISTORY = 100

# ========================
# Helper Functions
# ========================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load trained model and metadata"""
    global model, model_metadata, feature_extractor
    
    model_path = 'models/coffee_defect_model.pkl'
    metadata_path = 'models/model_metadata.json'
    
    try:
        # Load model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("[SUCCESS] Model loaded successfully")
        else:
            print("[WARN] Model file not found. Please train the model first.")
            return False
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print("[SUCCESS] Model metadata loaded successfully")
        else:
            print("[WARN] Metadata file not found")
            return False
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        return False

def process_image_file(file):
    """Process uploaded image file"""
    try:
        # Read image from file
        img = Image.open(file.stream)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Handle alpha channel and convert to BGR for OpenCV
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def get_image_preview(img_array):
    """Convert image to base64 for preview"""
    try:
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_display = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_display = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_display)
        
        # Compress for preview
        pil_img.thumbnail((400, 400))
        
        # Convert to base64
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_base64}"
    
    except Exception as e:
        print(f"Error creating preview: {str(e)}")
        return None

def predict_defect(img_array):
    """Predict coffee defect class"""
    try:
        if model is None:
            raise RuntimeError("Model not loaded")
        
        # Extract features
        features = feature_extractor.extract_features_from_image(img_array)
        features = features.reshape(1, -1)
        
        # Predict
        class_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class name and confidence
        class_name = model_metadata['class_names'][class_idx]
        confidence = float(probabilities[class_idx]) * 100
        
        # Get top 5 predictions
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        top_predictions = [
            {
                'class': model_metadata['class_names'][idx],
                'probability': round(float(probabilities[idx]) * 100, 2)
            }
            for idx in top_5_indices
        ]
        
        return {
            'predicted_class': class_name,
            'confidence': round(confidence, 2),
            'all_probabilities': {
                model_metadata['class_names'][i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(model_metadata['class_names']))
            },
            'top_predictions': top_predictions
        }
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

# ========================
# Routes - Pages
# ========================

@app.route('/')
def index():
    """Landing page / Home"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard / Classifier page"""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    """Analytics / Model Performance page"""
    return render_template('analytics.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# ========================
# Routes - API Endpoints
# ========================

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model not loaded'}), 404
        
        return jsonify({
            'status': 'success',
            'data': {
                'n_classes': model_metadata['training_history']['n_classes'],
                'n_features': model_metadata['training_history']['n_features'],
                'accuracy': round(model_metadata['training_history']['test_accuracy'] * 100, 2),
                'precision': round(model_metadata['training_history']['test_precision'] * 100, 2),
                'recall': round(model_metadata['training_history']['test_recall'] * 100, 2),
                'f1_score': round(model_metadata['training_history']['test_f1'] * 100, 2),
                'train_accuracy': round(model_metadata['training_history']['train_accuracy'] * 100, 2),
                'classes': model_metadata['class_names'],
                'n_training_samples': model_metadata['training_history']['n_train_samples'],
                'n_testing_samples': model_metadata['training_history']['n_test_samples'],
                'model_params': model_metadata.get('model_params', {}),
                'feature_extractor_info': model_metadata.get('feature_extractor_info', {})
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict coffee defect from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not available. Please train the model first.'
            }), 503
        
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed: jpg, jpeg, png, gif, bmp'
            }), 400
        
        # Process image
        img_array = process_image_file(file)
        
        # Get preview
        preview = get_image_preview(img_array)
        
        # Predict
        prediction_result = predict_defect(img_array)
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': secure_filename(file.filename),
            'predicted_class': prediction_result['predicted_class'],
            'confidence': prediction_result['confidence'],
            'preview': preview
        }
        prediction_history.append(history_entry)
        
        # Keep history size manageable
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': prediction_result,
                'image_preview': preview,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict from base64 image"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available'
            }), 503
        
        data = request.get_json()
        
        if 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image_data']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Convert RGB to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # Predict
        prediction_result = predict_defect(img_array)
        preview = get_image_preview(img_array)
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': 'uploaded_image',
            'predicted_class': prediction_result['predicted_class'],
            'confidence': prediction_result['confidence'],
            'preview': preview
        }
        prediction_history.append(history_entry)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': prediction_result,
                'image_preview': preview,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/api/prediction-history', methods=['GET'])
def get_prediction_history():
    """Get prediction history"""
    try:
        return jsonify({
            'status': 'success',
            'data': prediction_history[-20:]  # Return last 20 predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics from model and prediction history"""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model not available'}), 503
        
        # Class distribution from training data
        cm = np.array(model_metadata['training_history']['confusion_matrix'])
        row_sums = np.sum(cm, axis=1)
        # Avoid division by zero
        row_sums_safe = np.where(row_sums == 0, 1, row_sums)
        class_accuracy = np.diag(cm) / row_sums_safe
        
        # Prediction history stats
        if prediction_history:
            predicted_classes = [p['predicted_class'] for p in prediction_history]
            class_counts = {}
            for cls in predicted_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        else:
            class_counts = {}
        
        return jsonify({
            'status': 'success',
            'data': {
                'total_predictions': len(prediction_history),
                'class_accuracy': {
                    model_metadata['class_names'][i]: round(float(class_accuracy[i]) * 100, 2)
                    for i in range(len(model_metadata['class_names']))
                },
                'prediction_distribution': class_counts,
                'model_metrics': {
                    'accuracy': round(model_metadata['training_history']['test_accuracy'] * 100, 2),
                    'precision': round(model_metadata['training_history']['test_precision'] * 100, 2),
                    'recall': round(model_metadata['training_history']['test_recall'] * 100, 2),
                    'f1_score': round(model_metadata['training_history']['test_f1'] * 100, 2)
                }
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data for visualization"""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model not available'}), 503
        
        cm = model_metadata['training_history']['confusion_matrix']
        class_names = model_metadata['class_names']
        
        return jsonify({
            'status': 'success',
            'data': {
                'matrix': cm,
                'labels': class_names
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance data"""
    try:
        if model is None:
            return jsonify({'error': 'Model not available'}), 503
        
        importance = model.feature_importances_
        
        # Create feature labels
        feature_labels = []
        feature_labels.extend([f'HOG_{i}' for i in range(324)])
        feature_labels.extend([f'LBP_{i}' for i in range(59)])
        feature_labels.extend([f'ColorHist_{i}' for i in range(32)])
        feature_labels.extend([f'Edge_{i}' for i in range(33)])
        feature_labels.extend(['Morph_Area', 'Morph_Perimeter', 'Morph_Circularity'])
        
        # Ensure labels match feature count
        if len(feature_labels) != len(importance):
            feature_labels = [f'Feature_{i}' for i in range(len(importance))]
        
        # Get top 20 features
        top_indices = np.argsort(importance)[::-1][:20]
        top_features = [
            {
                'name': feature_labels[idx],
                'importance': round(float(importance[idx]), 6)
            }
            for idx in top_indices
        ]
        
        # Group importances by feature type
        groups = {
            'HOG': float(np.sum(importance[:324])),
            'LBP': float(np.sum(importance[324:383])),
            'Color Histogram': float(np.sum(importance[383:415])),
            'Edge': float(np.sum(importance[415:448])),
            'Morphological': float(np.sum(importance[448:]))
        }
        
        # Normalize groups
        total = sum(groups.values())
        if total > 0:
            groups = {k: round(v / total * 100, 2) for k, v in groups.items()}
        
        return jsonify({
            'status': 'success',
            'data': {
                'top_features': top_features,
                'feature_groups': groups
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/class-report', methods=['GET'])
def get_class_report():
    """Get per-class classification report"""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model not available'}), 503
        
        report = model_metadata['training_history'].get('classification_report', {})
        class_names = model_metadata['class_names']
        
        class_metrics = []
        for name in class_names:
            if name in report:
                class_metrics.append({
                    'class': name,
                    'precision': round(report[name].get('precision', 0) * 100, 2),
                    'recall': round(report[name].get('recall', 0) * 100, 2),
                    'f1_score': round(report[name].get('f1-score', 0) * 100, 2),
                    'support': report[name].get('support', 0)
                })
        
        return jsonify({
            'status': 'success',
            'data': class_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# ========================
# Error Handlers
# ========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ========================
# Startup
# ========================

@app.before_request
def before_request():
    """Executed before each request"""
    pass

@app.teardown_appcontext
def teardown(exception):
    """Cleanup after request"""
    pass

# ========================
# Main
# ========================

# Load model on import (for gunicorn)
print("[INFO] Initializing Coffee Defect Classifier...")
if load_model():
    print("[SUCCESS] Application ready!")
else:
    print("[WARN] Warning: Model not loaded. Some features may not work.")

if __name__ == '__main__':
    # Run app
    app.run(
        debug=os.environ.get('FLASK_DEBUG', False),
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )