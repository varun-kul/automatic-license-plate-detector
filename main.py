import torch
import torch.serialization
import threading
import time
from collections import deque
from torch.nn.modules.container import Sequential
from ultralytics.nn.tasks import DetectionModel

#torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential, 'ultralytics.nn.tasks.DetectionModel'])
torch.serialization.add_safe_globals([DetectionModel])

from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np
import re

def validate_indian_license_plate(text):
    """
    Validation specifically for Indian license plate formats
    """
    if not text or len(text) < 4:
        return False
    
    # Remove spaces and convert to uppercase
    clean_text = text.replace(' ', '').upper()
    
    # Check if it's alphanumeric
    if not re.match(r'^[A-Z0-9]+$', clean_text):
        return False
    
    # Indian license plates are typically 6-10 characters
    if len(clean_text) < 4 or len(clean_text) > 10:
        return False
    
    # Indian plates typically have this pattern:
    # Old format: XX## XX#### (e.g., MH01 AB1234)
    # New format: XX##XX#### (e.g., MH01AB1234) 
    # Where X = letter, # = number
    
    # Must have both letters and numbers
    has_letters = any(c.isalpha() for c in clean_text)
    has_numbers = any(c.isdigit() for c in clean_text)
    
    if not (has_letters and has_numbers):
        return False
    
    # Check common Indian state codes (partial list)
    indian_state_codes = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DN', 'DD', 'DL', 'GA', 
        'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 
        'ML', 'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 
        'UP', 'UK', 'WB', 'TG'
    ]
    
    # Check if it starts with a known state code (optional validation)
    # This is flexible - doesn't reject if state code not in list
    starts_with_state = any(clean_text.startswith(code) for code in indian_state_codes)
    
    # Additional pattern validation for Indian plates
    # Most Indian plates follow patterns like: XX##XXXX or XX##X#### 
    pattern_matches = [
        re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$', clean_text),  # Standard format
        re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', clean_text),      # Most common
        re.match(r'^[A-Z]{2}[0-9]{2}[A-Z][0-9]{1,4}$', clean_text),       # Variation
        re.match(r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}[0-9]{0,4}$', clean_text), # Flexible
    ]
    
    has_valid_pattern = any(pattern for pattern in pattern_matches)
    
    return has_valid_pattern or starts_with_state

def enhance_indian_license_plate(image):
    """
    Enhanced preprocessing specifically for Indian license plates
    """
    enhanced_images = []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small (Indian plates need good resolution)
    height, width = gray.shape
    if height < 60 or width < 200:
        scale_factor = max(60/height, 200/width)
        new_width = int(width * scale_factor * 1.5)
        new_height = int(height * scale_factor * 1.5)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    enhanced_images.append(("original_resized", gray))
    
    # 1. Aggressive denoising (Indian plates often have dirt/wear)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    enhanced_images.append(("denoised_aggressive", denoised))
    
    # 2. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    enhanced_images.append(("clahe_enhanced", clahe_img))
    
    # 3. Otsu thresholding (most reliable for real-time)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("otsu_thresh", otsu))
    
    # 4. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 2)
    enhanced_images.append(("adaptive_large", adaptive))
    
    return enhanced_images

class LiveLicensePlateDetector:
    def __init__(self):
        self.results = {}
        self.mot_tracker = Sort()
        
        # Load models
        print("Loading models...")
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('./models/best.pt')
        
        # Optimized settings for real-time processing
        self.license_plate_detector.conf = 0.2
        self.license_plate_detector.iou = 0.3
        
        self.vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # For tracking detected plates to avoid duplicates
        self.detected_plates = deque(maxlen=50)
        self.frame_count = 0
        
        # Initialize EasyOCR once
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized successfully")
        except ImportError:
            print("EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None
    
    def is_duplicate_plate(self, text):
        """Check if we've recently detected this plate"""
        current_time = time.time()
        # Remove old detections (older than 5 seconds)
        self.detected_plates = deque([
            (plate, timestamp) for plate, timestamp in self.detected_plates
            if current_time - timestamp < 5.0
        ], maxlen=50)
        
        # Check if this plate was recently detected
        for plate, timestamp in self.detected_plates:
            if plate == text and current_time - timestamp < 2.0:  # 2 second cooldown
                return True
        
        return False
    
    def add_detected_plate(self, text):
        """Add a newly detected plate to our tracking"""
        self.detected_plates.append((text, time.time()))
    
    def process_license_plate(self, license_plate_crop, detection_id):
        """Process a single license plate crop"""
        if self.reader is None:
            return None, 0, "no_ocr"
        
        # Use only the most reliable enhancement methods for real-time
        enhanced_images = enhance_indian_license_plate(license_plate_crop)
        
        best_text = None
        best_score = 0
        best_method = None
        
        # Try only the most effective methods for speed
        priority_methods = ["otsu_thresh", "clahe_enhanced", "original_resized"]
        
        for method_name, enhanced_img in enhanced_images:
            if method_name not in priority_methods:
                continue
                
            try:
                detections = self.reader.readtext(
                    enhanced_img,
                    width_ths=0.4,
                    height_ths=0.4,
                    paragraph=False,
                    detail=1,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                
                for detection in detections:
                    if len(detection) >= 3:
                        bbox, text, ocr_score = detection
                        
                        # Clean text
                        cleaned_text = text.upper().replace(' ', '').replace('-', '')
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', cleaned_text)
                        
                        if (cleaned_text and len(cleaned_text) >= 4 and 
                            validate_indian_license_plate(cleaned_text) and 
                            ocr_score > best_score):
                            best_text = cleaned_text
                            best_score = ocr_score
                            best_method = method_name
            
            except Exception as e:
                continue
        
        return best_text, best_score, best_method
    
    def process_frame(self, frame):
        """Process a single video frame"""
        self.frame_count += 1
        frame_results = {}
        
        # Detect vehicles
        detections = self.coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # Track vehicles
        track_ids = self.mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else np.array([])
        
        # Detect license plates
        license_plates = self.license_plate_detector(frame)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Skip low confidence detections
            if score < 0.15:
                continue
            
            # Assign license plate to car
            if len(track_ids) > 0:
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            else:
                car_id = len(frame_results) + 1
                xcar1, ycar1, xcar2, ycar2 = 0, 0, frame.shape[1], frame.shape[0]
            
            if car_id != -1:
                # Crop license plate with padding
                padding = 5
                y1_crop = max(0, int(y1) - padding)
                y2_crop = min(frame.shape[0], int(y2) + padding)
                x1_crop = max(0, int(x1) - padding)
                x2_crop = min(frame.shape[1], int(x2) + padding)
                
                license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop, :]
                
                # Skip if crop is too small
                if license_plate_crop.shape[0] < 10 or license_plate_crop.shape[1] < 30:
                    continue
                
                # Process license plate
                text, text_score, method = self.process_license_plate(
                    license_plate_crop, f"{self.frame_count}_{car_id}"
                )
                
                if text and not self.is_duplicate_plate(text):
                    self.add_detected_plate(text)
                    
                    frame_results[car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': text,
                            'bbox_score': score,
                            'text_score': text_score,
                            'detection_method': method
                        }
                    }
                    
                    print(f"Frame {self.frame_count}: Detected plate '{text}' (confidence: {text_score:.3f})")
        
        return frame_results, track_ids
    
    def draw_detections(self, frame, frame_results, track_ids):
        """Draw bounding boxes and text on the frame"""
        # Draw vehicle tracks
        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Vehicle {int(track_id)}', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw license plate detections
        for car_id, data in frame_results.items():
            # Draw license plate bbox
            x1, y1, x2, y2 = data['license_plate']['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Draw detected text
            text = data['license_plate']['text']
            confidence = data['license_plate']['text_score']
            cv2.putText(frame, f'{text} ({confidence:.2f})', 
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def run_live_detection(self, video_source=0):
        """Run live detection on video feed"""
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting live detection...")
        print("Press 'q' to quit, 's' to save current results")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                start_time = time.time()
                frame_results, track_ids = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Draw detections
                frame = self.draw_detections(frame, frame_results, track_ids)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    fps = fps_counter / (time.time() - fps_start_time) if time.time() - fps_start_time > 0 else 0
                
                # Add info overlay
                cv2.putText(frame, f'FPS: {fps:.1f} | Processing: {processing_time*1000:.1f}ms', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Frame: {self.frame_count} | Plates detected: {len(self.detected_plates)}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Store results
                if frame_results:
                    self.results[self.frame_count] = frame_results
                
                # Display frame
                cv2.imshow('Live Indian License Plate Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current results
                    if self.results:
                        write_csv(self.results, f'./live_detection_results_{int(time.time())}.csv')
                        print(f"Results saved! Total detections: {len(self.results)}")
                    else:
                        print("No detections to save")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Save final results
            if self.results:
                write_csv(self.results, f'./final_live_detection_results_{int(time.time())}.csv')
                print(f"\nFinal results saved! Total frames with detections: {len(self.results)}")
                
                # Print summary
                all_plates = set()
                for frame_data in self.results.values():
                    for car_data in frame_data.values():
                        all_plates.add(car_data['license_plate']['text'])
                
                print(f"Unique license plates detected: {len(all_plates)}")
                for plate in sorted(all_plates):
                    print(f"  - {plate}")

def main():
    """Main function to run live detection"""
    detector = LiveLicensePlateDetector()
    
    # You can specify different video sources:
    # 0 = default webcam
    # 1 = second camera
    # "path/to/video.mp4" = video file
    # "rtsp://..." = IP camera stream
    
    video_source = 0  # Change this as needed
    
    print("Indian License Plate Live Detection System")
    print("=========================================")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current results")
    print()
    
    detector.run_live_detection(video_source)

if __name__ == "__main__":
    main()