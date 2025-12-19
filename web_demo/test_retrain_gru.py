"""
Unit Tests for Model Training and Retraining

Tests the GRU4Rec training pipeline functionality:
- Data loading and preprocessing
- Training process
- Checkpoint management
- Evaluation metrics
"""

import pytest
import pandas as pd
import numpy as np
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

# Import functions from retrain_gru
import sys
sys.path.insert(0, os.path.dirname(__file__))
from retrain_gru import (
    load_logged_interactions,
    load_base_data,
    combine_data,
    train_gru,
    atomic_save
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_interactions_df():
    """Create sample interaction data."""
    return pd.DataFrame({
        'SessionId': [1, 1, 2, 2, 3, 3, 3],
        'ItemId': [10, 20, 15, 25, 10, 30, 40],
        'Time': [1000, 1005, 2000, 2010, 3000, 3005, 3010]
    })


@pytest.fixture
def sample_logged_csv(tmp_path):
    """Create temporary CSV with logged interactions."""
    csv_file = tmp_path / "interactions.csv"
    data = {
        'timestamp': ['2025-01-01 10:00:00', '2025-01-01 10:05:00', '2025-01-01 11:00:00'],
        'session': ['s1', 's1', 's2'],
        'item_id': ['100', '200', '150'],
        'event_type': ['click', 'add_to_cart', 'view_product'],
        'extra': ['', '', '']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def sample_base_data_file(tmp_path):
    """Create temporary base data file."""
    data_file = tmp_path / "base_data.csv"
    data = "1,100,1000\n1,200,1005\n2,150,2000\n"
    data_file.write_text(data)
    return str(data_file)


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving."""
    return str(tmp_path / "test_model.pt")


@pytest.fixture
def mock_gru4rec():
    """Create a mock GRU4Rec model."""
    mock = MagicMock()
    mock.fit = MagicMock()
    mock.savemodel = MagicMock()
    mock.data_iterator = MagicMock()
    mock.data_iterator.itemidmap = pd.Series([0, 1, 2], index=[100, 200, 150])
    return mock


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

class TestDataLoading:
    """Tests for data loading and preprocessing."""
    
    def test_load_logged_interactions_with_missing_file(self):
        """Verify handling of missing interactions file."""
        result = load_logged_interactions('/nonexistent/path.csv')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["SessionId", "ItemId", "Time"]
    
    def test_load_logged_interactions_filters_by_event_type(self, sample_logged_csv):
        """Verify only specified event types are loaded."""
        # The file has click, add_to_cart, view_product - all should be kept
        result = load_logged_interactions(sample_logged_csv, min_session_len=1)
        
        assert len(result) > 0
        assert 'SessionId' in result.columns
        assert 'ItemId' in result.columns
    
    def test_load_logged_interactions_filters_short_sessions(self, sample_logged_csv):
        """Verify sessions shorter than min_session_len are filtered."""
        result = load_logged_interactions(sample_logged_csv, min_session_len=5)
        
        # Should have fewer rows since sessions are short
        assert len(result) == 0  # All sessions have < 5 items
    
    def test_load_logged_interactions_returns_dataframe(self, sample_logged_csv):
        """Verify function returns DataFrame with correct columns."""
        result = load_logged_interactions(sample_logged_csv, min_session_len=1)
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert 'SessionId' in result.columns
            assert 'ItemId' in result.columns
            assert 'Time' in result.columns
    
    def test_load_base_data_with_missing_file(self):
        """Verify handling of missing base data file."""
        result = load_base_data('/nonexistent/base.csv')
        
        assert result is None
    
    def test_load_base_data_with_valid_file(self, sample_base_data_file):
        """Verify loading valid base data file."""
        result = load_base_data(sample_base_data_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'SessionId' in result.columns
        assert 'ItemId' in result.columns
        assert 'Time' in result.columns
    
    def test_load_base_data_converts_types(self, sample_base_data_file):
        """Verify data types are converted correctly."""
        result = load_base_data(sample_base_data_file)
        
        assert result['SessionId'].dtype == 'int64'
        assert result['ItemId'].dtype == 'int64'
        assert result['Time'].dtype == 'int64'
    
    def test_load_base_data_handles_milliseconds(self, tmp_path):
        """Verify conversion of millisecond timestamps to seconds."""
        data_file = tmp_path / "base_ms.csv"
        # Large timestamp values (milliseconds)
        data = "1,100,1609459200000\n1,200,1609459205000\n"
        data_file.write_text(data)
        
        result = load_base_data(str(data_file))
        
        # Should be converted to seconds
        assert result['Time'].max() < 10**11


# ============================================================================
# DATA COMBINATION TESTS
# ============================================================================

class TestDataCombination:
    """Tests for combining base and new data."""
    
    def test_combine_data_with_none_base(self, sample_interactions_df):
        """Verify combining when base_df is None."""
        result = combine_data(None, sample_interactions_df)
        
        assert result.equals(sample_interactions_df)
    
    def test_combine_data_with_empty_new(self, sample_interactions_df):
        """Verify combining when new_df is empty."""
        result = combine_data(sample_interactions_df, pd.DataFrame())
        
        assert result.equals(sample_interactions_df)
    
    def test_combine_data_with_both(self, sample_interactions_df):
        """Verify combining both base and new data."""
        new_df = pd.DataFrame({
            'SessionId': [4, 4],
            'ItemId': [50, 60],
            'Time': [4000, 4005]
        })
        
        result = combine_data(sample_interactions_df, new_df)
        
        assert len(result) == len(sample_interactions_df) + len(new_df)
        assert 'SessionId' in result.columns
    
    def test_combine_data_sorts_by_session_time(self, sample_interactions_df):
        """Verify combined data is sorted by session and time."""
        new_df = pd.DataFrame({
            'SessionId': [1, 1],
            'ItemId': [50, 60],
            'Time': [500, 600]  # Before existing session 1 items
        })
        
        result = combine_data(sample_interactions_df, new_df)
        
        # Check that for each session, times are in order
        for sid in result['SessionId'].unique():
            session_times = result[result['SessionId'] == sid]['Time'].values
            assert np.all(session_times[:-1] <= session_times[1:])
    
    def test_combine_data_removes_nulls(self):
        """Verify combining removes rows with null values."""
        base_df = pd.DataFrame({
            'SessionId': [1, 2],
            'ItemId': [10, None],
            'Time': [1000, 2000]
        })
        new_df = pd.DataFrame({
            'SessionId': [3],
            'ItemId': [30],
            'Time': [3000]
        })
        
        result = combine_data(base_df, new_df)
        
        # Should remove row with null ItemId
        assert not result.isnull().any().any()


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestModelTraining:
    """Tests for GRU4Rec training process."""
    
    def test_train_gru_with_empty_data(self):
        """Verify error when training with empty data."""
        empty_df = pd.DataFrame(columns=['SessionId', 'ItemId', 'Time'])
        
        with pytest.raises(ValueError, match="No data to train on"):
            train_gru(empty_df)
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_with_valid_data(self, mock_gru_class, sample_interactions_df):
        """Verify training succeeds with valid data."""
        mock_gru = MagicMock()
        mock_gru.fit = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        result = train_gru(sample_interactions_df, epochs=1, device='cpu')
        
        assert result is not None
        mock_gru.fit.assert_called_once()
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_uses_correct_epochs(self, mock_gru_class, sample_interactions_df):
        """Verify epochs parameter is passed correctly."""
        mock_gru = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        train_gru(sample_interactions_df, epochs=5, device='cpu')
        
        # Check that n_epochs in init matches
        call_kwargs = mock_gru_class.call_args[1]
        assert call_kwargs['n_epochs'] == 5
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_clamps_batch_size(self, mock_gru_class):
        """Verify batch size is clamped to number of sessions."""
        # Create data with only 2 sessions
        data_df = pd.DataFrame({
            'SessionId': [1, 1, 2],
            'ItemId': [10, 20, 30],
            'Time': [1000, 1005, 2000]
        })
        mock_gru = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        train_gru(data_df, epochs=1, device='cpu', params_override={'batch_size': 100})
        
        # Batch size should be clamped to 2 (number of sessions)
        call_kwargs = mock_gru_class.call_args[1]
        assert call_kwargs['batch_size'] <= 2
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_applies_param_overrides(self, mock_gru_class, sample_interactions_df):
        """Verify parameter overrides are applied."""
        mock_gru = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        train_gru(
            sample_interactions_df,
            epochs=1,
            device='cpu',
            params_override={'learning_rate': 0.01, 'dropout_p_embed': 0.5}
        )
        
        call_kwargs = mock_gru_class.call_args[1]
        assert call_kwargs['learning_rate'] == 0.01
        assert call_kwargs['dropout_p_embed'] == 0.5
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_device_parameter(self, mock_gru_class, sample_interactions_df):
        """Verify device parameter is passed correctly."""
        mock_gru = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        train_gru(sample_interactions_df, epochs=1, device='cuda')
        
        call_kwargs = mock_gru_class.call_args[1]
        assert call_kwargs['device'] == 'cuda'
    
    @patch('retrain_gru.GRU4Rec')
    def test_train_gru_returns_gru_model(self, mock_gru_class, sample_interactions_df):
        """Verify function returns trained GRU model."""
        mock_gru = MagicMock()
        mock_gru_class.return_value = mock_gru
        
        result = train_gru(sample_interactions_df, epochs=1)
        
        assert result == mock_gru


# ============================================================================
# CHECKPOINT MANAGEMENT TESTS
# ============================================================================

class TestCheckpointManagement:
    """Tests for model checkpoint saving and loading."""
    
    def test_atomic_save_creates_file(self, temp_model_path, mock_gru4rec):
        """Verify atomic_save creates model file."""
        atomic_save(mock_gru4rec, temp_model_path)
        
        # Verify savemodel was called
        mock_gru4rec.savemodel.assert_called_once()
    
    def test_atomic_save_creates_directory(self, tmp_path, mock_gru4rec):
        """Verify atomic_save creates parent directories if needed."""
        nested_path = str(tmp_path / "models" / "subdir" / "model.pt")
        
        atomic_save(mock_gru4rec, nested_path)
        
        mock_gru4rec.savemodel.assert_called_once()
        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_atomic_save_uses_temporary_file(self, temp_model_path, mock_gru4rec):
        """Verify atomic_save uses temporary file for safety."""
        with patch('tempfile.mkstemp') as mock_mkstemp:
            with patch('shutil.move'):
                with patch('os.close'):  # Mock os.close to prevent error
                    mock_fd = 100
                    mock_mkstemp.return_value = (mock_fd, '/tmp/tmpXXX.pt')
                    atomic_save(mock_gru4rec, temp_model_path)
                    mock_mkstemp.assert_called_once()
    
    def test_atomic_save_moves_file_atomically(self, temp_model_path, mock_gru4rec):
        """Verify atomic_save uses shutil.move for atomic operation."""
        with patch('tempfile.mkstemp') as mock_mkstemp:
            with patch('shutil.move') as mock_move:
                with patch('os.close'):  # Mock os.close to prevent error
                    mock_fd = 100
                    tmp_path = '/tmp/tmpXXX.pt'
                    mock_mkstemp.return_value = (mock_fd, tmp_path)
                
                    atomic_save(mock_gru4rec, temp_model_path)
                    
                    # Should move temp file to final path
                    mock_move.assert_called_once_with(tmp_path, temp_model_path)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_complete_training_pipeline(self, sample_interactions_df):
        """Verify complete training pipeline works end-to-end."""
        with patch('retrain_gru.GRU4Rec') as mock_gru_class:
            with patch('retrain_gru.atomic_save'):
                mock_gru = MagicMock()
                mock_gru_class.return_value = mock_gru
                
                # Train model
                trained_model = train_gru(sample_interactions_df, epochs=1)
                
                assert trained_model is not None
                mock_gru.fit.assert_called_once()
    
    def test_combine_and_train(self, sample_base_data_file):
        """Verify combining data and training works."""
        new_data = pd.DataFrame({
            'SessionId': [10, 10, 11],
            'ItemId': [100, 200, 150],
            'Time': [5000, 5005, 6000]
        })
        base_data = load_base_data(sample_base_data_file)
        
        with patch('retrain_gru.GRU4Rec') as mock_gru_class:
            mock_gru = MagicMock()
            mock_gru_class.return_value = mock_gru
            
            combined = combine_data(base_data, new_data)
            trained_model = train_gru(combined, epochs=1)
            
            assert trained_model is not None
            assert len(combined) == len(base_data) + len(new_data)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in training pipeline."""
    
    def test_load_base_data_handles_malformed_csv(self, tmp_path):
        """Verify handling of malformed CSV data."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("invalid,data,without,numbers")
        
        result = load_base_data(str(bad_file))
        
        # Should return empty DataFrame or None
        if result is not None:
            assert len(result) == 0
    
    def test_train_gru_handles_fit_error(self, sample_interactions_df):
        """Verify handling when GRU.fit() raises error."""
        with patch('retrain_gru.GRU4Rec') as mock_gru_class:
            mock_gru = MagicMock()
            mock_gru.fit.side_effect = RuntimeError("Fit failed")
            mock_gru_class.return_value = mock_gru
            
            with pytest.raises(RuntimeError):
                train_gru(sample_interactions_df, epochs=1)
    
    def test_atomic_save_handles_save_error(self, temp_model_path):
        """Verify handling when savemodel() fails."""
        mock_gru = MagicMock()
        mock_gru.savemodel.side_effect = IOError("Save failed")
        
        with pytest.raises(IOError):
            atomic_save(mock_gru, temp_model_path)
