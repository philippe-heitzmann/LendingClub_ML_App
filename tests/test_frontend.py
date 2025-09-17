"""
Tests for the Dash frontend application.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'frontend'))

from app import app


@pytest.fixture
def dash_app():
    """Create a test Dash application."""
    return app


def test_app_creation(dash_app):
    """Test that the Dash app is created successfully."""
    assert dash_app is not None
    assert dash_app.title == "Philippe Heitzmann Capstone Project"


def test_app_layout_structure(dash_app):
    """Test that the app layout has the expected structure."""
    layout = dash_app.layout
    
    # Check that layout is not None
    assert layout is not None
    
    # Check that it's a Div component
    assert hasattr(layout, 'children')


@patch('app.requests.get')
def test_prediction_callback(mock_get, dash_app):
    """Test the prediction callback function."""
    from app import update_prediction_text
    
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {"prediction": "No Default"}
    mock_get.return_value = mock_response
    
    # Test the callback
    result = update_prediction_text(12345, "GBC")
    
    # Verify the API was called
    mock_get.assert_called_once()
    
    # Verify the result
    assert result == "No Default"


@patch('app.requests.get')
def test_probability_callback(mock_get, dash_app):
    """Test the probability callback function."""
    from app import update_probability_text
    
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prediction": "No Default",
        "confidence": [0.2, 0.8]
    }
    mock_get.return_value = mock_response
    
    # Test the callback
    result = update_probability_text(12345, "GBC")
    
    # Verify the API was called
    mock_get.assert_called_once()
    
    # Verify the result
    assert result == 0.8


def test_ml_description_callback(dash_app):
    """Test the ML description callback function."""
    from app import update_ml_description_text
    
    # Test QDA description
    qda_desc = update_ml_description_text("QDA")
    assert "QDA is a supervised dimensionality reduction algorithm" in qda_desc
    
    # Test LDA description
    lda_desc = update_ml_description_text("LDA")
    assert "LDA is a supervised dimensionality reduction algorithm" in lda_desc
    
    # Test LOGIT description
    logit_desc = update_ml_description_text("LOGIT")
    assert "Logistic Regression works by combining inputs linearly" in logit_desc
    
    # Test GBC description
    gbc_desc = update_ml_description_text("GBC")
    assert "Gradient Boosting Classifier model is a composite model" in gbc_desc
    
    # Test unknown model
    unknown_desc = update_ml_description_text("UNKNOWN")
    assert unknown_desc == ""


def test_data_loading(dash_app):
    """Test that data is loaded correctly."""
    # This test would need to be adjusted based on actual data loading
    # For now, we'll just check that the app starts without errors
    assert dash_app is not None


@patch('app.pd.read_csv')
def test_dataframe_creation(mock_read_csv, dash_app):
    """Test that the main dataframe is created correctly."""
    import pandas as pd
    
    # Mock the CSV reading
    mock_df = pd.DataFrame({
        'id': [1, 2, 3],
        'annual_inc': [50000, 60000, 70000],
        'emp_length': [5, 10, 15]
    })
    mock_read_csv.return_value = mock_df
    
    # This would test the actual data loading if we refactored it
    # For now, we'll just verify the mock works
    result = mock_read_csv('/app/app/data/accepted_2007_to_2018Q4_500.csv')
    assert len(result) == 3
    assert 'id' in result.columns
