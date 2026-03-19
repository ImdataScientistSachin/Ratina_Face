import cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace
from typing import List, Dict, Any, Union

class FaceDetector:
    """
    A professional wrapper for detection and verification.
    """
    
    def __init__(self, threshold: float = 0.9, verification_model: str = "ArcFace"):
        self.threshold = threshold
        self.verification_model = verification_model

    # ... (keeping existing methods) ...

    def verify(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """
        Verify if two images contain the same person using DeepFace + RetinaFace.
        
        Args:
            img1_path: Path to the first image.
            img2_path: Path to the second image.
            
        Returns:
            A dictionary with verification results.
        """
        try:
            # We use detector_backend='retinaface' for better accuracy since it's already installed
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.verification_model,
                detector_backend='retinaface', 
                enforce_detection=False,
                align=True
            )
            return result
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"DeepFace Verification Error: {str(e)}")

    def detect(self, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect faces in an image.
        
        Args:
            image_input: Either a path to an image file (str) or a numpy array (OpenCV image).
            
        Returns:
            A dictionary where keys are face identifiers (e.g., 'face_1') and values 
            contain 'score', 'facial_area' [x1, y1, x2, y2], and 'landmarks'.
        """
        # RetinaFace.detect_faces handles both string paths and numpy arrays
        results = RetinaFace.detect_faces(image_input, threshold=self.threshold)
        
        # If no faces are found, RetinaFace might return an empty dict or None depending on version
        if not results:
            return {}
            
        return results

    def extract_faces(self, image_path: str, align: bool = True) -> List[np.ndarray]:
        """
        Extract and optionally align faces from an image path.
        
        Args:
            image_path: Path to the image file.
            align: Whether to perform facial alignment based on eye coordinates.
            
        Returns:
            A list of numpy arrays, each representing a cropped facial image.
        """
        faces = RetinaFace.extract_faces(img_path=image_path, align=align)
        return faces

    @staticmethod
    def draw_results(image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draws bounding boxes, landmarks, and text labels (Face ID, Score) on the image.
        """
        output_img = image.copy()
        
        for face_id, data in results.items():
            area = data.get("facial_area", [])
            score = data.get("score", 0.0)
            
            if len(area) == 4:
                # 1. Bounding Box
                cv2.rectangle(output_img, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 2)
                
                # 2. Text Label (Face ID and Confidence Score)
                label = f"{face_id} ({score:.2f})"
                cv2.putText(
                    output_img, label, (area[0], area[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # 3. Landmarks
            landmarks = data.get("landmarks", {})
            for point_name, coords in landmarks.items():
                cv2.circle(output_img, (int(coords[0]), int(coords[1])), 2, (0, 0, 255), -1)
                
        return output_img
