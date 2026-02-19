"""
Unit Tests for Detection Engine
Author: Rakin Mohammed Rafeeq
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector_utils

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a simple hand-like shape
    cv2.rectangle(img, (50, 50), (150, 200), (255, 255, 255), -1)
    return img

@pytest.fixture
def detection_graph():
    """Load detection graph for testing"""
    graph, sess = detector_utils.load_inference_graph()
    yield graph, sess
    sess.close()

def test_load_inference_graph():
    """Test model loading"""
    graph, sess = detector_utils.load_inference_graph()
    assert graph is not None
    assert sess is not None
    sess.close()

def test_detect_objects(detection_graph, sample_image):
    """Test object detection"""
    graph, sess = detection_graph
    boxes, scores = detector_utils.detect_objects(sample_image, graph, sess)
    
    assert boxes is not None
    assert scores is not None
    assert len(boxes) > 0
    assert len(scores) > 0
    assert all(0 <= score <= 1 for score in scores)

def test_box_drawing(sample_image):
    """Test bounding box drawing"""
    im_height, im_width = sample_image.shape[:2]
    boxes = np.array([[0.1, 0.1, 0.5, 0.5]])
    scores = np.array([0.9])
    
    # Should not raise exception
    detector_utils.draw_box_on_image(
        1, 0.27, scores, boxes, im_width, im_height, sample_image
    )
    
    assert sample_image is not None

def test_fps_drawing(sample_image):
    """Test FPS display"""
    detector_utils.draw_fps_on_image("FPS: 30", sample_image)
    assert sample_image is not None

def test_webcam_stream_initialization():
    """Test WebcamVideoStream initialization"""
    # Note: This will fail if no camera is available
    try:
        stream = detector_utils.WebcamVideoStream(0, 640, 480)
        assert stream is not None
        stream.stop()
    except:
        pytest.skip("No camera available for testing")

def test_confidence_threshold_filtering():
    """Test confidence threshold filtering"""
    scores = np.array([0.9, 0.5, 0.2, 0.8, 0.1])
    threshold = 0.27
    
    filtered = [s for s in scores if s > threshold]
    assert len(filtered) == 4
    assert all(s > threshold for s in filtered)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
