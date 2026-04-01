import cv2
import numpy as np
import time
import os
from retinaface import RetinaFace
from deepface import DeepFace
from typing import List, Dict, Any, Union

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class FaceDetector:
    """
    A professional wrapper for detection and verification with built-in 
    performance tracking and MLflow integration.
    """
    
    def __init__(self, threshold: float = 0.9, verification_model: str = "ArcFace", use_mlflow: bool = False):
        self.threshold = threshold
        self.verification_model = verification_model
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        if self.use_mlflow:
            # Set internal project experiment
            mlflow.set_experiment("RetinaFace_Inference_Monitoring")

    def verify(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """
        Verify if two images contain the same person using DeepFace + RetinaFace.
        """
        start_time = time.time()
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.verification_model,
                detector_backend='retinaface', 
                enforce_detection=False,
                align=True
            )
            
            latency = (time.time() - start_time) * 1000
            if self.use_mlflow:
                with mlflow.start_run(run_name="Verification_Task", nested=True):
                    mlflow.log_param("model", self.verification_model)
                    mlflow.log_metric("latency_ms", latency)
                    mlflow.log_metric("verified", 1 if result.get("verified") else 0)
                    mlflow.log_metric("distance", result.get("distance", 0))
            
            return result
        except Exception as e:
            raise Exception(f"DeepFace Verification Error: {str(e)}")

    def detect(self, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect faces in an image and track performance metrics.
        """
        start_time = time.time()
        
        # RetinaFace.detect_faces handles both string paths and numpy arrays
        results = RetinaFace.detect_faces(image_input, threshold=self.threshold)
        
        latency = (time.time() - start_time) * 1000
        
        if not results:
            if self.use_mlflow:
                self._log_detection_metrics(0, latency, 0)
            return {}
            
        num_faces = len(results)
        avg_confidence = np.mean([data.get("score", 0) for data in results.values()])
        
        if self.use_mlflow:
            self._log_detection_metrics(num_faces, latency, avg_confidence)
            
        return results

    def _log_detection_metrics(self, count: int, latency: float, confidence: float):
        """Helper to log inference metrics to MLflow."""
        try:
            with mlflow.start_run(run_name="Detection_Inference", nested=True):
                mlflow.log_param("threshold", self.threshold)
                mlflow.log_metric("face_count", count)
                mlflow.log_metric("latency_ms", latency)
                mlflow.log_metric("avg_confidence", confidence)
        except Exception as e:
            print(f"MLflow Logging Error: {e}")

    def extract_faces(self, image_path: str, align: bool = True) -> List[np.ndarray]:
        """
        Extract and optionally align faces from an image path.
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
                cv2.rectangle(output_img, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 2)
                label = f"{face_id} ({score:.2f})"
                cv2.putText(
                    output_img, label, (area[0], area[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            landmarks = data.get("landmarks", {})
            for point_name, coords in landmarks.items():
                cv2.circle(output_img, (int(coords[0]), int(coords[1])), 2, (0, 0, 255), -1)
                
        return output_img
