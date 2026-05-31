"""
Image Preprocessing & Feature Extraction Module
Modul untuk preprocessing gambar dan ekstraksi fitur untuk Random Forest
"""

import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class ImagePreprocessor:
    """Preprocessing gambar kopi dengan fitur-fitur optimal untuk Random Forest"""
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize preprocessor
        
        Args:
            img_size: Ukuran gambar output setelah resize
        """
        self.img_size = img_size
        self.scaler = StandardScaler()
    
    @staticmethod
    def load_image(image_path):
        """Load gambar dari file"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image from {image_path}")
        return img
    
    def preprocess_image(self, image_data):
        """
        Preprocess gambar:
        1. Load atau convert dari numpy array
        2. Convert ke RGB jika diperlukan
        3. Resize ke ukuran standar
        4. Convert ke Grayscale
        5. Normalisasi
        
        Args:
            image_data: Path atau numpy array
            
        Returns:
            Preprocessed grayscale image
        """
        # Load gambar
        if isinstance(image_data, str):
            img = self.load_image(image_data)
        else:
            img = image_data.copy()
        
        # Convert BGR to RGB jika dari OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize ke ukuran standar
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert ke grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Normalisasi (0-1)
        gray = gray.astype(np.float32) / 255.0
        
        # Histogram equalization untuk meningkatkan contrast
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_equalized = cv2.equalizeHist(gray_uint8)
        gray = gray_equalized.astype(np.float32) / 255.0
        
        return gray
    
    def extract_hog_features(self, gray_image):
        """
        Ekstrak HOG (Histogram of Oriented Gradients) features
        Excellent untuk mendeteksi shape dan texture
        """
        hog_features = feature.hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        return hog_features
    
    def extract_texture_features(self, gray_image):
        """
        Ekstrak texture features:
        - Contrast
        - Dissimilarity
        - Homogeneity
        - Energy
        """
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(gray_image, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
        
        return lbp_hist
    
    def extract_color_histogram(self, gray_image):
        """
        Ekstrak color histogram features dari grayscale
        256 bins untuk intensity levels
        """
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        
        # Ambil 32 bins yang di-downsample untuk mengurangi dimensionalitas
        hist_downsampled = hist[::8]  # Setiap 8 bins diambil 1
        
        return hist_downsampled
    
    def extract_edge_features(self, gray_image):
        """
        Ekstrak edge features menggunakan Canny edge detection
        """
        # Canny edge detection
        edges = cv2.Canny((gray_image * 255).astype(np.uint8), 100, 200)
        edges = edges.astype(np.float32) / 255.0
        
        # Hitung edge density
        edge_density = np.mean(edges)
        
        # Hitung edge histogram
        edge_hist, _ = np.histogram(edges, bins=32, range=(0, 1))
        edge_hist = edge_hist / np.sum(edge_hist)
        
        return np.concatenate([[edge_density], edge_hist])
    
    def extract_morphological_features(self, gray_image):
        """
        Ekstrak morphological features:
        - Area
        - Perimeter
        - Circularity
        """
        binary = ((gray_image > np.mean(gray_image)).astype(np.uint8) * 255)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            features.append(area / (self.img_size[0] * self.img_size[1]))  # Normalized area
            features.append(perimeter / (2 * (self.img_size[0] + self.img_size[1])))  # Normalized perimeter
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features.append(circularity)
            else:
                features.append(0)
        else:
            features = [0, 0, 0]
        
        return np.array(features)
    
    def extract_all_features(self, image_data):
        """
        Ekstrak semua fitur sekaligus
        
        Returns:
            Feature vector (1D array)
        """
        # Preprocess gambar
        gray_image = self.preprocess_image(image_data)
        
        # Ekstrak berbagai jenis fitur
        hog_feat = self.extract_hog_features(gray_image)
        lbp_feat = self.extract_texture_features(gray_image)
        color_feat = self.extract_color_histogram(gray_image)
        edge_feat = self.extract_edge_features(gray_image)
        morph_feat = self.extract_morphological_features(gray_image)
        
        # Gabungkan semua fitur
        all_features = np.concatenate([
            hog_feat,  # ~324 features
            lbp_feat,  # 59 features
            color_feat,  # 32 features
            edge_feat,  # 33 features
            morph_feat  # 3 features
        ])
        
        return all_features
    
    def normalize_features(self, features):
        """Normalize fitur menggunakan StandardScaler"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.scaler.fit_transform(features)


class FeatureExtractor:
    """High-level feature extractor untuk digunakan dalam training"""
    
    def __init__(self, img_size=(128, 128)):
        self.preprocessor = ImagePreprocessor(img_size)
    
    def extract_features_from_image(self, image_path_or_array):
        """Extract features dari single image"""
        return self.preprocessor.extract_all_features(image_path_or_array)
    
    def extract_batch_features(self, image_list):
        """Extract features dari batch gambar"""
        features = []
        for img_path in image_list:
            try:
                feat = self.extract_features_from_image(img_path)
                features.append(feat)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        return np.array(features)


# Feature dimensionality summary:
# HOG: 324 (9 orientations * 16 cells * 2x2 blocks)
# LBP: 59 (uniform patterns)
# Color Histogram: 32 (downsampled from 256)
# Edge Features: 33 (1 density + 32 histogram bins)
# Morphological: 3 (area, perimeter, circularity)
# TOTAL: ~451 features untuk Random Forest
