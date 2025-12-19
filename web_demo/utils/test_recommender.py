"""
Unit Tests for GRURecommender Engine

Tests the recommender system's core functionality:
- Model initialization
- Recommendation generation
- Session state management
- Error handling
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from recommender import GRURecommender


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_gru_model():
    """Create a mock GRU4Rec model."""
    mock = MagicMock()
    mock.model = MagicMock()
    mock.model.eval = MagicMock()
    mock.model.layers = [16, 16]  # Two layers with hidden size 16
    mock.device = 'cpu'
    mock.data_iterator = MagicMock()
    
    # Create itemidmap: item_id -> item_idx
    itemidmap = pd.Series([0, 1, 2, 3, 4, 5], index=['item_1', 'item_2', 'item_3', 'item_4', 'item_5', 'item_6'])
    mock.data_iterator.itemidmap = itemidmap
    
    return mock


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary mock model file."""
    model_file = tmp_path / "test_model.pt"
    model_file.write_text("mock model")
    return str(model_file)


@pytest.fixture
def recommender_with_mock(mock_gru_model, mock_model_path):
    """Create a GRURecommender with mocked GRU4Rec."""
    with patch('recommender.GRU4Rec') as mock_gru4rec_class:
        with patch('recommender.os.path.isfile', return_value=True):
            mock_gru4rec_class.loadmodel.return_value = mock_gru_model
            recommender = GRURecommender(mock_model_path)
            recommender.gru = mock_gru_model
    return recommender


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestGRURecommenderInitialization:
    """Tests for recommender initialization."""
    
    def test_initialization_with_nonexistent_model(self):
        """Verify error when model file doesn't exist."""
        recommender = GRURecommender('/nonexistent/path/model.pt')
        assert not recommender.ok
        assert "Model file not found" in recommender.error
    
    def test_initialization_with_valid_model(self, recommender_with_mock, mock_gru_model):
        """Verify successful initialization with valid model."""
        assert recommender_with_mock.ok
        assert recommender_with_mock.gru is not None
        assert recommender_with_mock.itemidmap is not None
        assert recommender_with_mock.idx_to_item is not None
    
    def test_initialization_creates_idx_to_item_mapping(self, recommender_with_mock):
        """Verify idx_to_item mapping is created correctly."""
        assert len(recommender_with_mock.idx_to_item) > 0
        # idx_to_item should be inverse of itemidmap
        for item_id, idx in recommender_with_mock.itemidmap.items():
            assert recommender_with_mock.idx_to_item[idx] == item_id
    
    def test_initialization_creates_popular_items_fallback(self, recommender_with_mock):
        """Verify popular items fallback list is created."""
        assert hasattr(recommender_with_mock, 'popular_items')
        assert len(recommender_with_mock.popular_items) > 0
    
    def test_initialization_creates_session_states_dict(self, recommender_with_mock):
        """Verify session_states dictionary is initialized."""
        assert isinstance(recommender_with_mock.session_states, dict)
        assert len(recommender_with_mock.session_states) == 0
    
    def test_initialization_device_parameter(self, mock_gru_model, mock_model_path):
        """Verify device parameter is passed correctly."""
        with patch('recommender.GRU4Rec') as mock_gru4rec_class:
            with patch('recommender.os.path.isfile', return_value=True):
                mock_gru4rec_class.loadmodel.return_value = mock_gru_model
                recommender = GRURecommender(mock_model_path, device='cuda')
                mock_gru4rec_class.loadmodel.assert_called_with(mock_model_path, device='cuda')
    
    def test_initialization_with_empty_itemidmap(self, mock_gru_model, mock_model_path):
        """Verify error when itemidmap is empty."""
        mock_gru_model.data_iterator.itemidmap = pd.Series([], dtype=object)
        with patch('recommender.GRU4Rec') as mock_gru4rec_class:
            with patch('recommender.os.path.isfile', return_value=True):
                mock_gru4rec_class.loadmodel.return_value = mock_gru_model
                recommender = GRURecommender(mock_model_path)
                assert not recommender.ok
                assert "itemidmap empty" in recommender.error


# ============================================================================
# RECOMMENDATION GENERATION TESTS
# ============================================================================

class TestRecommendationGeneration:
    """Tests for recommendation generation."""
    
    def test_recommend_with_single_item_session(self, recommender_with_mock, mock_gru_model):
        """Verify recommendations for session with single item."""
        # Mock the model's scoring function
        mock_scores = np.array([0.5, 0.8, 0.3, 0.9, 0.2, 0.1])
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1'], topk=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 3
        assert all(isinstance(item, str) for item in recommendations)
    
    def test_recommend_with_multiple_items_session(self, recommender_with_mock, mock_gru_model):
        """Verify recommendations for session with multiple items."""
        mock_scores = np.array([0.1, 0.8, 0.3, 0.9, 0.7, 0.2])
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1', 'item_2'], topk=4)
        
        assert len(recommendations) == 4
        assert all(item in ['item_1', 'item_2', 'item_3', 'item_4', 'item_5', 'item_6'] 
                   for item in recommendations)
    
    def test_recommend_respects_topk_parameter(self, recommender_with_mock, mock_gru_model):
        """Verify topk parameter limits number of recommendations."""
        mock_scores = np.random.rand(6)
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        for k in [1, 3, 5]:
            recommendations = recommender_with_mock.recommend(['item_1'], topk=k)
            assert len(recommendations) == k
    
    def test_recommend_excludes_seen_items_by_default(self, recommender_with_mock, mock_gru_model):
        """Verify seen items are excluded from recommendations by default."""
        mock_scores = np.ones(6) * 0.5
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1', 'item_2'], topk=4, exclude_seen=True)  # Ask for 4 instead
        
        assert 'item_1' not in recommendations
        assert 'item_2' not in recommendations
        assert len(recommendations) == 4  # Verify we get exactly 4 non-excluded items
    
    def test_recommend_includes_seen_when_disabled(self, recommender_with_mock, mock_gru_model):
        """Verify seen items can be included when exclude_seen=False."""
        # Set high scores for seen items
        mock_scores = np.array([0.9, 0.95, 0.3, 0.2, 0.1, 0.05])
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1', 'item_2'], topk=3, exclude_seen=False)
        
        assert 'item_1' in recommendations or 'item_2' in recommendations
    
    def test_recommend_with_empty_session(self, recommender_with_mock):
        """Verify recommendations for empty session return popular items."""
        recommendations = recommender_with_mock.recommend([], topk=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all(isinstance(item, str) for item in recommendations)
    
    def test_recommend_returns_empty_when_not_ok(self, mock_gru_model, mock_model_path):
        """Verify empty recommendations when recommender not initialized."""
        with patch('recommender.os.path.isfile', return_value=False):
            recommender = GRURecommender(mock_model_path)
            assert not recommender.ok
            
            recommendations = recommender.recommend(['item_1'])
            assert recommendations == []
    
    def test_recommend_handles_invalid_item_ids(self, recommender_with_mock, mock_gru_model):
        """Verify handling of item IDs not in itemidmap."""
        mock_scores = np.random.rand(6)
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        # Pass mix of valid and invalid items
        recommendations = recommender_with_mock.recommend(['item_1', 'invalid_item', 'item_2'], topk=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
    
    def test_recommend_no_duplicates(self, recommender_with_mock, mock_gru_model):
        """Verify recommendations don't contain duplicates."""
        mock_scores = np.random.rand(6)
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1'], topk=6)
        
        assert len(recommendations) == len(set(recommendations))
    
    def test_recommend_returns_valid_item_ids(self, recommender_with_mock, mock_gru_model):
        """Verify all recommendations are valid item IDs."""
        mock_scores = np.random.rand(6)
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        valid_items = set(recommender_with_mock.itemidmap.index)
        recommendations = recommender_with_mock.recommend(['item_1'], topk=6)
        
        for item in recommendations:
            assert item in valid_items


# ============================================================================
# SESSION STATE MANAGEMENT TESTS
# ============================================================================

class TestSessionStateManagement:
    """Tests for session state tracking and updates."""
    
    def test_update_session_with_valid_item(self, recommender_with_mock, mock_gru_model):
        """Verify session state is updated with valid item."""
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        session_id = "session_1"
        recommender_with_mock.update_session(session_id, "item_1")
        
        assert session_id in recommender_with_mock.session_states
        assert len(recommender_with_mock.session_states[session_id]) == 2  # Two GRU layers
    
    def test_update_session_with_invalid_item(self, recommender_with_mock):
        """Verify update_session handles invalid item gracefully."""
        session_id = "session_1"
        # Should not raise error
        recommender_with_mock.update_session(session_id, "invalid_item")
        
        # Session state should not be created for invalid item
        assert session_id not in recommender_with_mock.session_states
    
    def test_update_session_accumulates_state(self, recommender_with_mock, mock_gru_model):
        """Verify sequential updates accumulate session state."""
        import copy
        
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.ones((1, 16)) * 0.5)
        
        session_id = "session_1"
        recommender_with_mock.update_session(session_id, "item_1")
        state_after_first = copy.deepcopy(recommender_with_mock.session_states[session_id])  # Deep copy
        
        # Change mock to return different value on next call
        mock_gru_model.model.hidden_step = Mock(return_value=torch.ones((1, 16)) * 0.7)  # Different value
        
        recommender_with_mock.update_session(session_id, "item_2")
        state_after_second = copy.deepcopy(recommender_with_mock.session_states[session_id])  # Deep copy
        
        # H[-1] (last hidden state) should be different
        assert not torch.allclose(state_after_first[-1], state_after_second[-1])  # Check last element, not first
    
    def test_reset_session_clears_state(self, recommender_with_mock, mock_gru_model):
        """Verify reset_session clears session state."""
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        session_id = "session_1"
        recommender_with_mock.update_session(session_id, "item_1")
        assert session_id in recommender_with_mock.session_states
        
        recommender_with_mock.reset_session(session_id)
        assert session_id not in recommender_with_mock.session_states
    
    def test_reset_session_with_none_session_id(self, recommender_with_mock):
        """Verify reset_session handles None session_id gracefully."""
        # Should not raise error
        recommender_with_mock.reset_session(None)
    
    def test_multiple_concurrent_sessions(self, recommender_with_mock, mock_gru_model):
        """Verify handling of multiple concurrent sessions."""
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        session_ids = ["session_1", "session_2", "session_3"]
        for i, session_id in enumerate(session_ids):
            recommender_with_mock.update_session(session_id, f"item_{i+1}")
        
        assert len(recommender_with_mock.session_states) == 3
        for session_id in session_ids:
            assert session_id in recommender_with_mock.session_states
    
    def test_recommend_uses_existing_session_state(self, recommender_with_mock, mock_gru_model):
        """Verify recommend uses stored session state when available."""
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        mock_scores = np.array([0.1, 0.8, 0.3, 0.9, 0.7, 0.2])
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.Wy = Mock()
        mock_gru_model.model.Wy.weight = Mock()
        mock_gru_model.model.By = Mock()
        mock_gru_model.model.By.weight = Mock()
        
        session_id = "session_1"
        recommender_with_mock.update_session(session_id, "item_1")
        
        # Recommend with session_id should use stored state
        recommendations = recommender_with_mock.recommend(['item_1'], session_id=session_id, topk=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3


# ============================================================================
# ERROR HANDLING & EDGE CASES
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""
    
    def test_recommend_with_topk_larger_than_vocab(self, recommender_with_mock, mock_gru_model):
        """Verify handling when topk exceeds vocabulary size."""
        vocab_size = len(recommender_with_mock.itemidmap)
        mock_scores = np.random.rand(vocab_size)
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1'], topk=100)
        
        # Should return at most vocab_size items
        assert len(recommendations) <= vocab_size
    
    def test_recommendation_scores_are_ranked(self, recommender_with_mock, mock_gru_model):
        """Verify recommendations are sorted by score in descending order."""
        mock_scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.4])
        mock_gru_model.model.score_items = Mock(return_value=torch.tensor(mock_scores))
        mock_gru_model.model.embed = Mock(return_value=(torch.zeros(1, 16), None, None))
        mock_gru_model.model.hidden_step = Mock(return_value=torch.zeros((1, 16)))
        
        recommendations = recommender_with_mock.recommend(['item_1'], topk=3, exclude_seen=False)
        
        # Top 3 indices should be: 1 (0.9), 3 (0.8), 5 (0.4)
        expected_top_3 = ['item_2', 'item_4', 'item_6']
        assert recommendations == expected_top_3
    
    def test_reload_method_reinitializes(self, mock_gru_model, mock_model_path):
        """Verify reload method reinitializes the recommender."""
        with patch('recommender.GRU4Rec') as mock_gru4rec_class:
            with patch('recommender.os.path.isfile', return_value=True):
                mock_gru4rec_class.loadmodel.return_value = mock_gru_model
                
                recommender = GRURecommender(mock_model_path)
                initial_ok = recommender.ok
                
                recommender.reload(mock_model_path)
                
                # After reload, ok status should still be the same (in this case True)
                assert recommender.ok == initial_ok
    
    def test_itemidmap_string_conversion(self, recommender_with_mock):
        """Verify item IDs are properly converted to strings."""
        recommendations = recommender_with_mock.recommend([], topk=3)
        
        for item in recommendations:
            assert isinstance(item, str)
    
    def test_graceful_fallback_on_model_error(self, recommender_with_mock, mock_gru_model):
        """Verify graceful fallback to popular items on model error."""
        # Make model.hidden_step raise an error
        mock_gru_model.model.embed = Mock(side_effect=RuntimeError("Model error"))
        
        recommendations = recommender_with_mock.recommend(['item_1'], topk=3)
        
        # Should fall back to popular items
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
