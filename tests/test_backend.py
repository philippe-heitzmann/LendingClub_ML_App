"""
Tests for the Flask backend API.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'backend'))

from flask_serve import app, models


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_default_endpoint(client):
    """Test the default endpoint returns expected message."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Lending Club dataset-trained machine learning models' in response.data


def test_predict_endpoint_missing_data(client):
    """Test predict endpoint with missing data returns 400."""
    response = client.get('/api/v1/predict')
    assert response.status_code == 400


def test_predict_endpoint_invalid_model(client):
    """Test predict endpoint with invalid model returns error."""
    data = {
        "query": [[1, 2, 3, 4]],
        "model": "INVALID_MODEL"
    }
    response = client.get('/api/v1/predict', json=data)
    assert response.status_code == 200
    # Should return None for invalid model
    assert response.json is None


@patch('flask_serve.models')
def test_predict_endpoint_valid_request(mock_models, client):
    """Test predict endpoint with valid request."""
    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_models.get.return_value = mock_model
    
    data = {
        "query": [[50000, 700, 5, 10]],
        "model": "GBC"
    }
    
    response = client.get('/api/v1/predict', json=data)
    assert response.status_code == 200
    
    response_data = response.json
    assert response_data['prediction'] == 'No Default'
    assert response_data['confidence'] == [0.2, 0.8]


@patch('flask_serve.models')
def test_predict_endpoint_default_prediction(mock_models, client):
    """Test predict endpoint returns 'Default' for prediction 0."""
    # Mock the model to return 0 (default)
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    mock_models.get.return_value = mock_model
    
    data = {
        "query": [[50000, 600, 2, 15]],
        "model": "LOGIT"
    }
    
    response = client.get('/api/v1/predict', json=data)
    assert response.status_code == 200
    
    response_data = response.json
    assert response_data['prediction'] == 'Default'
    assert response_data['confidence'] == [0.8, 0.2]


def test_models_loaded():
    """Test that models are properly loaded."""
    assert 'QDA' in models
    assert 'LDA' in models
    assert 'LOGIT' in models
    assert 'GBC' in models
    assert len(models) == 4
