import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.api.main import app, handler

client = TestClient(app)

# Replace the handler's predict method with a mock for testing
handler.predict = MagicMock()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Plant Disease Classification API is running"}

def test_health_healthy():
    # Mock that model is loaded
    handler.model = MagicMock()
    handler.model_name = "test_model"
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "test_model"}

def test_health_unhealthy():
    # Mock that model is not loaded
    handler.model = None
    
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}

def test_predict_not_an_image():
    # Create a non-image file
    file_content = b"This is a text file"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}
    
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert response.json() == {"detail": "File provided is not an image"}

def test_predict_success():
    # Mock the predict method
    handler.predict.return_value = {
        "class_id": 1,
        "confidence": 0.95,
        "class_name": "Apple___Apple_scab"
    }
    
    # Create a dummy image file
    file_content = b"fake image bytes"
    files = {"file": ("test.jpg", BytesIO(file_content), "image/jpeg")}
    
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    assert response.json() == {
        "class_id": 1,
        "confidence": 0.95,
        "class_name": "Apple___Apple_scab"
    }

def test_predict_internal_error():
    # Mock the predict method to return an error dict
    handler.predict.return_value = {
        "error": "Model prediction failed"
    }
    
    # Create a dummy image file
    file_content = b"fake image bytes"
    files = {"file": ("test.jpg", BytesIO(file_content), "image/jpeg")}
    
    response = client.post("/predict", files=files)
    assert response.status_code == 500
    assert response.json() == {"detail": "500: Model prediction failed"}

def test_predict_exception():
    # Mock the predict method to raise an exception
    handler.predict.side_effect = Exception("Unexpected error")
    
    # Create a dummy image file
    file_content = b"fake image bytes"
    files = {"file": ("test.jpg", BytesIO(file_content), "image/jpeg")}
    
    response = client.post("/predict", files=files)
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected error"}
