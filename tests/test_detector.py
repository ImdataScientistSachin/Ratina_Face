import pytest
import cv2
import numpy as np
import os
from src.detector import FaceDetector

# --- Fixtures ---

@pytest.fixture
def detector():
    """Provides an initialized FaceDetector instance."""
    return FaceDetector(threshold=0.9)

@pytest.fixture
def sample_image_path():
    """Provides the path to a sample image if it exists."""
    # Try multiple common names from the examples folder
    examples = ["examples/Amber.jpg", "examples/A_Heard.jpg", "examples/celeb_selfee.jpeg"]
    for img in examples:
        if os.path.exists(img):
            return img
    pytest.skip("No sample images found in examples/ directory for testing.")

# --- Tests ---

def test_detector_initialization(detector):
    """Verify that the detector class can be instantiated."""
    assert detector.threshold == 0.9

def test_detect_faces_path(detector, sample_image_path):
    """Verify that the detector returns a dictionary of results."""
    results = detector.detect(sample_image_path)
    assert isinstance(results, dict)
    # Since we use images with faces, we expect at least one face
    assert len(results) >= 1
    
    # Check if first face has required fields
    face_1 = results.get("face_1")
    assert face_1 is not None
    assert "facial_area" in face_1
    assert "landmarks" in face_1

def test_detect_faces_array(detector, sample_image_path):
    """Verify that the detector can handle numpy arrays."""
    img = cv2.imread(sample_image_path)
    results = detector.detect(img)
    assert isinstance(results, dict)
    assert len(results) >= 1

def test_extract_faces(detector, sample_image_path):
    """Verify that face extraction returns a list of crops."""
    faces = detector.extract_faces(sample_image_path, align=True)
    assert isinstance(faces, list)
    assert len(faces) >= 1
    # Each crop should be a numpy array
    assert isinstance(faces[0], np.ndarray)

def test_verify_same_person(detector, sample_image_path):
    """Verify identity match for the same image."""
    result = detector.verify(sample_image_path, sample_image_path)
    assert isinstance(result, dict)
    assert result["verified"] is True
    assert "distance" in result
    assert result["model"] == "ArcFace"

def test_draw_results(detector, sample_image_path):
    """Verify that visualization helper returns a valid image."""
    img = cv2.imread(sample_image_path)
    results = detector.detect(img)
    annotated = detector.draw_results(img, results)
    
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    # Check if image data is different (some pixels modified)
    assert not np.array_equal(img, annotated)
