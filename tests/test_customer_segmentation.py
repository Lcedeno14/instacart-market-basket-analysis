import pytest
import pandas as pd
import numpy as np
from src.analysis.customer_segmentation import (
    calculate_rfm,
    perform_clustering,
    analyze_department_preferences,
    analyze_purchase_patterns,
    create_segmentation_visualization
)

@pytest.fixture
def sample_order_data():
    """Create sample order data for testing."""
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'order_id': [1, 2, 3, 1, 2, 1, 2, 3, 4],
        'product_id': [101, 102, 103, 201, 202, 301, 302, 303, 304],
        'days_since_prior_order': [np.nan, 7, 14, np.nan, 10, np.nan, 5, 5, 5],
        'department': ['Dairy', 'Produce', 'Bakery', 'Dairy', 'Produce', 
                      'Dairy', 'Produce', 'Bakery', 'Dairy'],
        'order_hour_of_day': [9, 14, 10, 16, 11, 8, 13, 15, 12],
        'day_of_week': ['Monday', 'Wednesday', 'Friday', 'Tuesday', 'Thursday',
                       'Monday', 'Wednesday', 'Friday', 'Saturday']
    })

def test_calculate_rfm(sample_order_data):
    """Test RFM calculation."""
    rfm = calculate_rfm(sample_order_data)
    
    assert isinstance(rfm, pd.DataFrame)
    assert all(col in rfm.columns for col in ['frequency', 'monetary', 'recency'])
    assert len(rfm) == 3  # 3 unique users
    assert rfm.loc[1, 'frequency'] == 3  # User 1 has 3 orders
    assert rfm.loc[1, 'monetary'] == 3  # User 1 has 3 products

def test_perform_clustering(sample_order_data):
    """Test clustering functionality."""
    rfm = calculate_rfm(sample_order_data)
    clustered = perform_clustering(rfm, n_clusters=2)
    
    assert isinstance(clustered, pd.DataFrame)
    assert 'cluster' in clustered.columns
    assert 'cluster_size' in clustered.columns
    assert 'cluster_label' in clustered.columns
    assert clustered['cluster'].nunique() == 2

def test_analyze_department_preferences(sample_order_data):
    """Test department preference analysis."""
    dept_prefs = analyze_department_preferences(sample_order_data, n_clusters=2)
    
    assert isinstance(dept_prefs, pd.DataFrame)
    assert 'cluster' in dept_prefs.columns
    assert all(dept in dept_prefs.columns for dept in ['Dairy', 'Produce', 'Bakery'])
    assert dept_prefs['cluster'].nunique() == 2

def test_analyze_purchase_patterns(sample_order_data):
    """Test purchase pattern analysis."""
    patterns = analyze_purchase_patterns(sample_order_data, n_clusters=2, sample_size=3)
    
    assert isinstance(patterns, pd.DataFrame)
    assert 'cluster' in patterns.columns
    assert patterns['cluster'].nunique() == 2

def test_create_segmentation_visualization(sample_order_data):
    """Test visualization creation."""
    # Test RFM visualization
    rfm = calculate_rfm(sample_order_data)
    clustered = perform_clustering(rfm, n_clusters=2)
    fig_rfm = create_segmentation_visualization(clustered, 'rfm', 2)
    assert fig_rfm is not None
    
    # Test department preferences visualization
    dept_prefs = analyze_department_preferences(sample_order_data, n_clusters=2)
    fig_dept = create_segmentation_visualization(dept_prefs, 'departments', 2)
    assert fig_dept is not None
    
    # Test purchase patterns visualization
    patterns = analyze_purchase_patterns(sample_order_data, n_clusters=2)
    fig_patterns = create_segmentation_visualization(patterns, 'patterns', 2)
    assert fig_patterns is not None

def test_error_handling():
    """Test error handling with invalid data."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    assert isinstance(calculate_rfm(empty_df), pd.DataFrame)
    assert len(calculate_rfm(empty_df)) == 0
    
    # Test with invalid data
    invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
    assert isinstance(calculate_rfm(invalid_df), pd.DataFrame)
    assert len(calculate_rfm(invalid_df)) == 0

def test_edge_cases(sample_order_data):
    """Test edge cases in segmentation analysis."""
    # Test with single cluster
    rfm = calculate_rfm(sample_order_data)
    clustered = perform_clustering(rfm, n_clusters=1)
    assert clustered['cluster'].nunique() == 1
    
    # Test with more clusters than users
    with pytest.raises(Exception):
        perform_clustering(rfm, n_clusters=10)
    
    # Test with zero clusters
    with pytest.raises(Exception):
        perform_clustering(rfm, n_clusters=0) 