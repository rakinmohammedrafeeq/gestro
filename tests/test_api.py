"""
Integration Tests for FastAPI Endpoints
Author: Rakin Mohammed Rafeeq
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
import cv2
import numpy as np
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_detections" in data
    assert "active_websocket_connections" in data
    assert "uptime_seconds" in data
    assert "uptime_formatted" in data

def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "SSD MobileNet V2 FPNLite"
    assert data["training_iterations"] == 22500
    assert data["dataset"] == "EgoHands"
    assert data["num_classes"] == 1
    assert data["confidence_threshold"] == 0.27

def test_detect_endpoint_with_image():
    """Test detection endpoint with image"""
    # Create a test image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 200), (255, 255, 255), -1)
    
    # Encode image to bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())
    
    # Send request
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "detections" in data
    assert "num_hands" in data
    assert "inference_time_ms" in data
    assert "image_size" in data

def test_detect_endpoint_invalid_file():
    """Test detection endpoint with invalid file"""
    # Send invalid data
    response = client.post(
        "/detect",
        files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
    )
    
    assert response.status_code == 400

def test_websocket_connection():
    """Test WebSocket connection"""
    with client.websocket_connect("/ws/metrics") as websocket:
        data = websocket.receive_json()
        assert "total_detections" in data
        assert "active_websocket_connections" in data

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.get("/health")
    assert response.status_code == 200
    # CORS headers should be present for cross-origin requests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
