"""
Improved Iris Extractor with Robust Segmentation and Daugman Normalization
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class IrisExtractorV3:
    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        self.normalized_width = 512
        self.normalized_height = 64
        
    def segment_iris_robust(self, image):
        """
        Robust iris segmentation using edge detection + Hough circles.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Gaussian blur for smoother detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        
        # Try multiple threshold values to find circles
        best_circles = None
        best_score = -1
        
        for threshold1 in [30, 40, 50]:
            for threshold2 in [20, 30, 40]:
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=80,
                    param1=threshold1,
                    param2=threshold2,
                    minRadius=30,
                    maxRadius=140
                )
                
                if circles is not None and circles.shape[2] >= 2:
                    circles_data = circles[0, :, :]
                    # Sort by radius
                    circles_data = circles_data[np.argsort(circles_data[:, 2])]
                    
                    pupil_r = circles_data[0, 2]
                    iris_r = circles_data[1, 2] if len(circles_data) > 1 else circles_data[0, 2] * 1.5
                    
                    # Check if separation is good
                    if 25 <= pupil_r <= 100 and iris_r > pupil_r * 1.3 and iris_r <= 180:
                        score = iris_r - pupil_r  # Prefer good separation
                        if score > best_score:
                            best_score = score
                            best_circles = circles_data
        
        if best_circles is None or len(best_circles) < 2:
            return None
        
        pupil_x, pupil_y, pupil_r = best_circles[0]
        iris_x, iris_y, iris_r = best_circles[1] if len(best_circles) > 1 else best_circles[0]
        
        # Quality check
        if iris_r <= pupil_r:
            return None
        
        return {
            'pupil': (int(pupil_x), int(pupil_y), int(pupil_r)),
            'iris': (int(iris_x), int(iris_y), int(iris_r))
        }
    
    def normalize_iris_daugman(self, image, iris_circle, pupil_circle):
        """
        Proper Daugman rubber sheet normalization.
        """
        iris_x, iris_y, iris_r = iris_circle
        pupil_x, pupil_y, pupil_r = pupil_circle
        
        normalized = np.zeros((self.normalized_height, self.normalized_width), dtype=np.uint8)
        mask = np.zeros((self.normalized_height, self.normalized_width), dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # For each point in normalized space
        for ny in range(self.normalized_height):
            for nx in range(self.normalized_width):
                # Radius fraction within iris
                r_frac = ny / self.normalized_height
                
                # Angle in radians
                theta = 2 * np.pi * nx / self.normalized_width
                
                # Compute point on pupil boundary at this angle
                x_pupil = pupil_x + pupil_r * np.cos(theta)
                y_pupil = pupil_y + pupil_r * np.sin(theta)
                
                # Compute point on iris boundary at this angle
                x_iris = iris_x + iris_r * np.cos(theta)
                y_iris = iris_y + iris_r * np.sin(theta)
                
                # Interpolate
                x_norm = x_pupil + (x_iris - x_pupil) * r_frac
                y_norm = y_pupil + (y_iris - y_pupil) * r_frac
                
                x_int = int(np.round(x_norm))
                y_int = int(np.round(y_norm))
                
                if 0 <= x_int < w and 0 <= y_int < h:
                    normalized[ny, nx] = image[y_int, x_int]
                    mask[ny, nx] = 1
        
        return normalized, mask
    
    def extract_iris_code_log_gabor(self, normalized_iris, mask):
        """
        Extract iris code using Log-Gabor filters.
        """
        h, w = normalized_iris.shape
        
        # Normalize image
        iris_norm = normalized_iris.astype(np.float32)
        iris_norm = (iris_norm - iris_norm.mean()) / (iris_norm.std() + 1e-7)
        
        iris_code = np.zeros(2048, dtype=np.uint8)
        code_idx = 0
        
        # 4 scales × 8 orientations = 32 filters × 64 bits = 2048 bits
        scales = [4, 6, 8, 10]
        orientations = 8
        samples_per_filter = 64
        
        for scale_idx, scale in enumerate(scales):
            for ori_idx in range(orientations):
                # Log-Gabor parameters
                wavelength = scale
                theta = ori_idx * np.pi / orientations
                
                # Create filter
                log_gabor = self._log_gabor_filter(h, w, wavelength, theta)
                
                # Apply filter
                response = cv2.filter2D(iris_norm, -1, 
                                      np.real(log_gabor).astype(np.float32))
                
                # Binarize and sample
                response_bits = (response > response.mean()).astype(np.uint8)
                sampled_bits = response_bits.flatten()[::max(1, len(response_bits.flatten()) // samples_per_filter)][:samples_per_filter]
                
                # Insert into code
                for bit in sampled_bits:
                    if code_idx < 2048:
                        iris_code[code_idx] = bit
                        code_idx += 1
        
        return iris_code
    
    def _log_gabor_filter(self, height, width, wavelength, orientation):
        """
        Create a Log-Gabor filter in frequency domain.
        """
        # Frequency grid
        freq_x = np.fft.fftfreq(width)[:, np.newaxis]
        freq_y = np.fft.fftfreq(height)[np.newaxis, :]
        
        # Radial frequency
        freq = np.sqrt(freq_x**2 + freq_y**2) + 1e-6
        
        # Angular frequency
        angle = np.arctan2(freq_y, freq_x)
        
        # Log-Gabor filter components
        wavelength_inv = 1.0 / wavelength
        radial = np.exp(-0.5 * (np.log(freq / wavelength_inv)**2) / (0.5**2))
        
        # Cosine angular window
        angular_std = np.pi / 8
        angular = np.exp(-0.5 * ((angle - orientation)**2) / (angular_std**2))
        
        # Avoid division by zero
        angular[freq < 0.01] = 0
        
        # Convective Log-Gabor
        log_gabor = radial * angular * np.exp(1j * 2 * np.pi * wavelength_inv * freq)
        
        return log_gabor
    
    def process_single_image(self, image_path):
        """Complete extraction pipeline."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Segment
        seg = self.segment_iris_robust(gray)
        if seg is None:
            return None
        
        # Normalize  
        normalized, mask = self.normalize_iris_daugman(gray, seg['iris'], seg['pupil'])
        
        # Extract code
        iris_code = self.extract_iris_code_log_gabor(normalized, mask)
        
        return iris_code
    
    def process_casia_database(self, input_dir, output_dir):
        """Process entire database."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.bmp")))
        
        print(f"Found {len(image_paths)} images to process...")
        
        success = 0
        failed = 0
        
        for idx, img_path in enumerate(image_paths, 1):
            filename = img_path.stem  # e.g., "S1001L01"
            
            # Map to output format: user_session
            # S1001L01 -> user=1, session=01
            if filename.startswith('S1') and len(filename) >= 7:
                user_id = filename[1:4]  # "001"
                session = filename[-2:]   # "01"
                
                # Convert user ID to simple number (001 -> 1, 002 -> 2, etc.)
                user_num = str(int(user_id))
                output_file = output_path / f"{user_num}_{session}.txt"
            else:
                continue
            
            if output_file.exists():
                continue
            
            iris_code = self.process_single_image(img_path)
            
            if iris_code is not None:
                with open(output_file, 'w') as f:
                    f.write(','.join(map(str, iris_code)))
                success += 1
            else:
                failed += 1
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(image_paths)} (Success: {success}, Failed: {failed})")
        
        print(f"\nExtraction Complete!")
        print(f"Success: {success}, Failed: {failed}")
        return success, failed

def main():
    extractor = IrisExtractorV3()
    input_dir = "../../../data/Iris/CASIA-Iris-Interval"
    output_dir = "../../../data/Iris/Extracted_Codes"
    
    success, failed = extractor.process_casia_database(input_dir, output_dir)
    print(f"\nExtracted {success} Iris codes")

if __name__ == "__main__":
    main()
