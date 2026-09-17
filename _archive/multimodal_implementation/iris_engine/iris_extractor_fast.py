"""
Fast Iris Extractor - Simplified Log-Gabor in Spatial Domain
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class IrisExtractorFast:
    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        self.normalized_width = 512
        self.normalized_height = 64
        
    def segment_iris_robust(self, image):
        """Fast iris segmentation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        
        # Hough circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
            param1=40, param2=30, minRadius=30, maxRadius=140
        )
        
        if circles is None or circles.shape[2] < 2:
            return None
        
        circles = circles[0, :]
        circles = circles[np.argsort(circles[:, 2])]
        
        pupil_x, pupil_y, pupil_r = circles[0]
        iris_x, iris_y, iris_r = circles[1]
        
        if iris_r <= pupil_r or pupil_r < 20 or iris_r > 180:
            return None
        
        return {
            'pupil': (int(pupil_x), int(pupil_y), int(pupil_r)),
            'iris': (int(iris_x), int(iris_y), int(iris_r))
        }
    
    def normalize_iris(self, image, iris_circle, pupil_circle):
        """Daugman normalization."""
        iris_x, iris_y, iris_r = iris_circle
        pupil_x, pupil_y, pupil_r = pupil_circle
        
        norm = np.zeros((self.normalized_height, self.normalized_width), dtype=np.uint8)
        h, w = image.shape[:2]
        
        for ny in range(self.normalized_height):
            r_frac = ny / self.normalized_height
            for nx in range(self.normalized_width):
                theta = 2 * np.pi * nx / self.normalized_width
                
                x_p = pupil_x + pupil_r * np.cos(theta)
                y_p = pupil_y + pupil_r * np.sin(theta)
                x_i = iris_x + iris_r * np.cos(theta)
                y_i = iris_y + iris_r * np.sin(theta)
                
                x = x_p + (x_i - x_p) * r_frac
                y = y_p + (y_i - y_p) * r_frac
                
                xi = int(np.round(x))
                yi = int(np.round(y))
                
                if 0 <= xi < w and 0 <= yi < h:
                    norm[ny, nx] = image[yi, xi]
        
        return norm
    
    def extract_iris_code_fast(self, normalized):
        """Fast iris code extraction using simple Gabor-like filters."""
        h, w = normalized.shape
        iris_norm = (normalized.astype(np.float32) - normalized.mean()) / (normalized.std() + 1e-7)
        
        iris_code = np.zeros(2048, dtype=np.uint8)
        code_idx = 0
        
        # 4 scales × 8 orientations with simple kernels
        wavelengths = [8, 12, 16, 20]
        orientations = 8
        
        for wv_idx, wavelength in enumerate(wavelengths):
            for ori_idx in range(orientations):
                # Create simple oriented Gabor kernel
                kernel = self._create_simple_gabor(wavelength, orientations, ori_idx)
                
                # Convolve
                response = cv2.filter2D(iris_norm, -1, kernel)
                
                # Binarize
                bits = (response > response.mean()).astype(np.uint8)
                sampled = bits.flatten()[::max(1, len(bits.flatten()) // 64)][:64]
                
                iris_code[code_idx:code_idx+len(sampled)] = sampled
                code_idx += len(sampled)
        
        return iris_code[:2048]
    
    def _create_simple_gabor(self, wavelength, num_or, or_idx):
        """Create simple Gabor kernel."""
        size = int(wavelength * 2) | 1
        angle = or_idx * np.pi / num_or
        
        x = np.arange(-size//2, size//2 + 1)
        X, Y = np.meshgrid(x, x)
        
        Xtheta = X * np.cos(angle) + Y * np.sin(angle)
        Ytheta = -X * np.sin(angle) + Y * np.cos(angle)
        
        sigma = wavelength / 3
        envelope = np.exp(-(Xtheta**2 + Ytheta**2) / (2 * sigma**2))
        carrier = np.cos(2 * np.pi * Xtheta / wavelength)
        
        gabor = envelope * carrier
        return (gabor / (gabor.sum() + 1e-7)).astype(np.float32)
    
    def process_single_image(self, image_path):
        """Complete pipeline."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image = cv2.resize(image, self.image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        seg = self.segment_iris_robust(gray)
        if seg is None:
            return None
        
        normalized = self.normalize_iris(gray, seg['iris'], seg['pupil'])
        iris_code = self.extract_iris_code_fast(normalized)
        
        return iris_code
    
    def process_casia_database(self, input_dir, output_dir):
        """Process database."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.bmp")))
        
        print(f"Found {len(image_paths)} images")
        success, failed = 0, 0
        
        for idx, img_path in enumerate(image_paths, 1):
            filename = img_path.stem
            
            if filename.startswith('S1') and len(filename) >= 7:
                user_id = str(int(filename[1:4]))
                session = filename[-2:]
                output_file = output_path / f"{user_id}_{session}.txt"
            else:
                continue
            
            if output_file.exists():
                continue
            
            iris_code = self.process_single_image(str(img_path))
            
            if iris_code is not None:
                with open(output_file, 'w') as f:
                    f.write(','.join(map(str, iris_code)))
                success += 1
            else:
                failed += 1
            
            if idx % 200 == 0:
                print(f"Processed {idx}/{len(image_paths)} (Success: {success}, Failed: {failed})")
        
        print(f"\nComplete! Success: {success}, Failed: {failed}")
        return success, failed

if __name__ == "__main__":
    e = IrisExtractorFast()
    e.process_casia_database("../../../data/Iris/CASIA-Iris-Interval", "../../../data/Iris/Extracted_Codes")
