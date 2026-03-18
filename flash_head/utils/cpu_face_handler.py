import cv2
import numpy as np
from typing import Tuple, List

class CPUFaceHandler:
    """Handler for CPU-based face detection using OpenCV Haar Cascades."""

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.0):
        """Initialize the face detection handler."""
        # Use OpenCV's built-in Haar cascade for frontal face
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, image: np.ndarray) -> Tuple[int, List[int]]:
        """Detect faces in the given image.

        Args:
            image (np.ndarray): RGB image array.

        Returns:
            Tuple[int, List[int]]: A tuple containing:
                - Number of faces detected (int)
                - Bounding box coordinates [x1, y1, x2, y2]
        """
        bboxs, scores = [], []
        # Convert RGB to grayscale for OpenCV face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Sort faces by area (width * height) in descending order to get the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        for (x, y, w, h) in faces:
            # Add to list, convert to relative coordinates to match original mediapipe logic?
            # Wait, the original code returned relative coordinates!
            # Let's check original code: x, y, w, h = bboxC.xmin, ... x1, y1, x2, y2 = x, y, x + w, y + h
            # bboxC was relative_bounding_box
            img_h, img_w = image.shape[:2]
            
            x1_rel = x / img_w
            y1_rel = y / img_h
            x2_rel = (x + w) / img_w
            y2_rel = (y + h) / img_h
            
            bboxs.append([x1_rel, y1_rel, x2_rel, y2_rel])
            scores.append(1.0) # Dummy score
            
            # Only return the largest face
            break
            
        return bboxs, scores

    def __call__(self, image: np.ndarray) -> Tuple[int, List[int]]:
        return self.detect(image)
