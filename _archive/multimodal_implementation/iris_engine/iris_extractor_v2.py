"""
Production-Grade Iris Code Extractor using Daugman's Rubber Sheet Model
and Multi-Scale Log-Gabor Filtering.

Implementation based on:
- Daugman (2004): "How iris recognition works"
- Masek & Kovesi (2003): "MATLAB Source Code for Pupil Detection and Iris Segmentation"
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings('ignore')

class IrisExtractor:
    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        self.normalized_width = 512  # Daugman standard
        self.normalized_height = 64
        
    def segment_iris(self, image):
        """
        Segment iris region using multi-scale Hough circles.
        Returns: pupil circle, iris circle, quality score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect circles using multi-scale Hough (pupil and iris)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=150
        )
        
        if circles is None or circles.shape[2] < 2:
            return None
        
        circles = np.uint16(np.around(circles))
        circles = circles[0, :, :]
        
        # Sort by radius - pupil is smaller, iris is larger
        circles = circles[np.argsort(circles[:, 2])]
        
        # Pupil is the smallest circle
        pupil_x, pupil_y, pupil_r = circles[0]
        
        # Iris is typically the next circle (larger)
        if len(circles) >= 2:
            iris_x, iris_y, iris_r = circles[1]
        else:
            # If only one circle, estimate iris as ~1.5x radius
            iris_x, iris_y, iris_r = pupil_x, pupil_y, int(pupil_r * 1.5)
        
        # Sanity checks
        if iris_r <= pupil_r or pupil_r < 15 or iris_r > 200:
            return None
        
        # Quality check: contrast within iris region
        iris_region = blurred[max(0, iris_y - iris_r):min(blurred.shape[0], iris_y + iris_r),
                              max(0, iris_x - iris_r):min(blurred.shape[1], iris_x + iris_r)]
        
        if iris_region.size == 0:
            return None
        
        contrast = iris_region.std()
        if contrast < 10:  # Too low contrast = poor quality
            return None
        
        return {
            'pupil': (pupil_x, pupil_y, pupil_r),
            'iris': (iris_x, iris_y, iris_r),
            'contrast': contrast,
            'quality': contrast / 100.0  # Normalize quality score
        }
    
    def mask_eyelids_eyelashes(self, image, iris_circle, pupil_circle):
        """
        Create a mask to exclude eyelids and eyelashes.
        Returns binary mask where 1 = valid iris region, 0 = excluded
        """
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        
        iris_x, iris_y, iris_r = iris_circle
        pupil_x, pupil_y, pupil_r = pupil_circle
        
        # Create circular mask for valid iris region (annulus)
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Distance from iris center
        dist_from_iris = np.sqrt((x_coords - iris_x)**2 + (y_coords - iris_y)**2)
        
        # Valid: within iris circle but outside pupil
        valid_iris = (dist_from_iris <= iris_r) & (dist_from_iris >= pupil_r)
        mask[~valid_iris] = 0
        
        # Exclude top and bottom (eyelids)
        eyelid_margin = int(iris_r * 0.35)
        mask[iris_y - iris_r:iris_y - iris_r + eyelid_margin, :] = 0
        mask[iris_y + iris_r - eyelid_margin:iris_y + iris_r, :] = 0
        
        # Morphological closing to connect small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def normalize_iris_daugman(self, image, iris_circle, pupil_circle, mask):
        """
        Apply Daugman's Rubber Sheet Model: unwrap iris region to fixed dimensions.
        Maps from (r, θ) in polar coords to (x, y) in normalized Cartesian space.
        """
        iris_x, iris_y, iris_r = iris_circle
        pupil_x, pupil_y, pupil_r = pupil_circle
        
        normalized = np.zeros((self.normalized_height, self.normalized_width), dtype=np.uint8)
        
        # For each point in normalized space, find corresponding point in original image
        for ny in range(self.normalized_height):
            # Radius fraction: 0 (pupil boundary) to 1 (iris boundary)
            r_fraction = (pupil_r + (iris_r - pupil_r) * ny / self.normalized_height) / iris_r
            
            for nx in range(self.normalized_width):
                # Angle in radians
                angle = 2 * np.pi * nx / self.normalized_width
                
                # Polar coordinates in original image
                # Point on iris boundary at this angle
                x = iris_x + iris_r * np.cos(angle)
                y = iris_y + iris_r * np.sin(angle)
                
                # Point on pupil boundary at this angle  
                x0 = pupil_x + pupil_r * np.cos(angle)
                y0 = pupil_y + pupil_r * np.sin(angle)
                
                # Interpolate between pupil and iris boundary
                x_interp = x0 + (x - x0) * r_fraction
                y_interp = y0 + (y - y0) * r_fraction
                
                # Bilinear interpolation from original image
                x_int, y_int = int(np.round(x_interp)), int(np.round(y_interp))
                
                if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
                    normalized[ny, nx] = image[y_int, x_int]
                else:
                    normalized[ny, nx] = 0
        
        return normalized
    
    def create_log_gabor_filters(self, height, width, num_scales=4, num_orientations=8):
        """
        Create multi-scale Log-Gabor filters.
        Scales: 4 (typical for iris)
        Orientations: 8 (0, π/8, π/4, 3π/8, π/2, 5π/8, 3π/4, 7π/8)
        """
        filters = []
        
        min_wavelength = 4
        mult = 1.3  # Scaling factor between filters
        
        for scale in range(num_scales):
            wavelength = min_wavelength * (mult ** scale)
            sigma = wavelength * 0.56  # Standard deviation
            
            for orientation in range(num_orientations):
                theta = np.pi * orientation / num_orientations
                
                # Create filter grid
                y_grid, x_grid = np.ogrid[-height//2:height//2, -width//2:width//2]
                
                # Rotate coordinates
                x_theta = x_grid * np.cos(theta) + y_grid * np.sin(theta)
                y_theta = -x_grid * np.sin(theta) + y_grid * np.cos(theta)
                
                # Log-Gabor response
                r = np.sqrt(x_theta**2 + y_theta**2)
                r[r == 0] = 1  # Avoid log(0)
                
                # Radial and angular components
                log_gabor = np.exp(-0.5 * (np.log(r / wavelength))**2 / (np.log(sigma / wavelength))**2)
                angular = np.exp(-0.5 * (y_theta**2 / (sigma**2)))
                
                # Gabor filter
                gabor = log_gabor * angular * np.exp(1j * 2 * np.pi * x_theta / wavelength)
                
                filters.append(gabor)
        
        return filters
    
    def extract_iris_code(self, normalized_iris):
        """
        Extract binary iris code using multi-scale Log-Gabor filtering.
        Returns 2048-bit code (4 scales * 8 orientations * 64 samples).
        """
        height, width = normalized_iris.shape
        
        # Create Log-Gabor filters
        filters = self.create_log_gabor_filters(height, width, num_scales=4, num_orientations=8)
        
        # Normalize iris image
        iris_normalized = (normalized_iris - normalized_iris.mean()) / (normalized_iris.std() + 1e-6)
        
        iris_code = []
        
        for gabor_filter in filters:
            # Convolve with iris image
            response = cv2.filter2D(iris_normalized.astype(np.float32), -1, np.real(gabor_filter).astype(np.float32))
            
            # Binarize: positive response = 1, negative = 0
            code_bits = (response > 0).astype(np.uint8).flatten()
            
            # Take samples (downsample to 64 bits per filter for manageable size)
            sampled = code_bits[::max(1, len(code_bits) // 64)][:64]
            iris_code.extend(sampled)
        
        # Pad to exactly 2048 bits
        iris_code = np.array(iris_code[:2048], dtype=np.uint8)
        if len(iris_code) < 2048:
            iris_code = np.pad(iris_code, (0, 2048 - len(iris_code)), mode='constant')
        
        return iris_code
    
    def process_single_image(self, image_path):
        """
        Complete pipeline: segment -> normalize -> extract code.
        Returns iris code or None if extraction fails.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize to standard size
        image = cv2.resize(image, self.image_size)
        
        # Segment iris
        segmentation = self.segment_iris(image)
        if segmentation is None:
            return None
        
        iris_circle = segmentation['iris']
        pupil_circle = segmentation['pupil']
        
        # Extract grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create mask
        mask = self.mask_eyelids_eyelashes(gray, iris_circle, pupil_circle)
        
        # Normalize
        normalized = self.normalize_iris_daugman(gray, iris_circle, pupil_circle, mask)
        
        # Extract code
        iris_code = self.extract_iris_code(normalized)
        
        return iris_code
    
    def process_casia_database(self, input_dir, output_dir):
        """
        Process entire CASIA-Iris-Interval database.
        Input: /path/to/CASIA-Iris-Interval/
        Output: /path/to/Extracted_Codes/ with files like 1_1.txt, 1_2.txt, etc.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all iris images
        image_paths = sorted(input_path.glob("**/*.bmp")) + sorted(input_path.glob("**/*.jpg"))
        
        print(f"Found {len(image_paths)} images to process...")
        
        success_count = 0
        fail_count = 0
        
        for idx, img_path in enumerate(image_paths, 1):
            # Extract user_id and session info from filename
            # CASIA format: xxx_y_z_t.bmp where xxx=ID, y=eye, z=session, t=image
            filename = img_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                user_id = parts[0]
                session = parts[2]
                code_filename = f"{user_id}_{session}.txt"
            else:
                continue
            
            output_file = output_path / code_filename
            
            # Skip if already processed
            if output_file.exists():
                continue
            
            # Extract iris code
            iris_code = self.process_single_image(img_path)
            
            if iris_code is not None:
                # Save as comma-separated binary values
                with open(output_file, 'w') as f:
                    f.write(','.join(map(str, iris_code)))
                success_count += 1
            else:
                fail_count += 1
            
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(image_paths)} (Success: {success_count}, Failed: {fail_count})")
        
        print(f"\nExtraction Complete!")
        print(f"Success: {success_count}, Failed: {fail_count}")
        return success_count, fail_count

def main():
    extractor = IrisExtractor()
    input_dir = "../../../data/Iris/CASIA-Iris-Interval"
    output_dir = "../../../data/Iris/Extracted_Codes"
    
    success, failed = extractor.process_casia_database(input_dir, output_dir)
    print(f"\nExtracted {success} Iris codes to {output_dir}")

if __name__ == "__main__":
    main()
