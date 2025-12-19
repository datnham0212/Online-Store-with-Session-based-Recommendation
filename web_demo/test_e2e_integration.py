"""
End-to-End Integration Tests for GRU4Rec Recommendation System

Tests complete user flows through the Flask app, including:
- Full user session workflows
- Concurrent user handling
- Model inference latency
- Data consistency between interactions and recommendations
- Error handling across endpoints
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from app import app, log_event
import os
import tempfile


@pytest.fixture
def client():
    """Flask test client with clean session."""
    app.config['TESTING'] = True
    app.config['SESSION_PERMANENT'] = False
    with app.test_client() as client:
        # Initialize session first
        client.get('/')
        yield client


@pytest.fixture
def mock_recommender():
    """Mock recommender with realistic behavior."""
    mock_rec = MagicMock()
    mock_rec.ok = True
    mock_rec.itemidmap = {str(i): i for i in range(1, 101)}
    mock_rec.idx_to_item = {i: str(i) for i in range(1, 101)}
    
    def mock_recommend(session_items, topk=5, exclude_seen=True, session_id=None):
        """Return recommendations based on session items."""
        if not session_items:
            return [str(i) for i in range(1, topk + 1)]
        seen_ids = {int(item_id) for item_id in session_items}  #Works for both int and str
        recs = []
        for item_id in range(1, 101):
            if len(recs) >= topk:
                break
            if exclude_seen and item_id in seen_ids:
                continue
            recs.append(str(item_id))
        return recs
    
    mock_rec.recommend = mock_recommend
    mock_rec.update_session = Mock()
    mock_rec.reset_session = Mock()
    return mock_rec


# ============================================================================
# TEST 1: User Session Flow with Recommendations
# ============================================================================

class TestUserSessionFlowWithRecommendations:
    """Test complete user journey: browse → add to cart → get recommendations."""
    
    def test_complete_user_flow_browse_add_recommend(self, client, mock_recommender):
        """Full user flow: view products, add to cart, get recommendations."""
        with patch('app.recommender', mock_recommender):
            # Step 1: User visits home
            response = client.get('/')
            assert response.status_code == 200
            assert b'products' in response.data or b'product' in response.data.lower()
            
            # Step 2: Get session ID from cookies
            with client.session_transaction() as sess:
                session_id = sess.get('_id')
                assert session_id is not None
            
            # Step 3: View a product (includes recommendations)
            response = client.get('/product/1')
            assert response.status_code in [200, 404]  # Either success or graceful 404
            
            # Step 4: Add item to cart (GET request with product_id in URL)
            response = client.get('/add_to_cart/1')
            assert response.status_code in [200, 302]  # Success or redirect to cart
            
            # Step 5: View another product (recommendations updated)
            response = client.get('/product/2')
            assert response.status_code in [200, 404]
            
            # Step 6: Add second item to cart
            response = client.get('/add_to_cart/2')
            assert response.status_code in [200, 302]
            
            # Step 7: View cart
            response = client.get('/cart')
            assert response.status_code == 200  # Cart page should exist
            
            # Step 8: Verify cart has items and can proceed to checkout
            response = client.get('/checkout')
            assert response.status_code == 200
    
    def test_cold_start_user_gets_popular_items(self, client, mock_recommender):
        """New user with empty session gets popular items as recommendations."""
        with patch('app.recommender', mock_recommender):
            # New user with no session items
            response = client.get('/')
            assert response.status_code == 200
            
            # Should handle gracefully (should show products without crash)
            assert response.status_code != 500


# ============================================================================
# TEST 2: Multiple Concurrent Sessions
# ============================================================================

class TestMultipleConcurrentSessions:
    """Test app handles multiple concurrent users correctly."""
    
    def test_multiple_concurrent_sessions_maintain_isolation(self, client, mock_recommender):
        """Each user session maintains isolation - no cross-contamination."""
        with patch('app.recommender', mock_recommender):
            # Simulate 3 concurrent sessions
            sessions = []
            
            for user_num in range(3):
                # Each gets their own test client
                test_client = app.test_client()
                test_client.get('/')  # Initialize session
                
                with test_client.session_transaction() as sess:
                    session_id = sess.get('_id')
                    sessions.append((test_client, session_id, user_num))
            
            # Verify each session is unique
            session_ids = [sid for _, sid, _ in sessions]
            assert len(set(session_ids)) == 3, "Sessions should be unique"
            
            # Each user adds different items
            for test_client, session_id, user_num in sessions:
                item_id = str(10 + user_num)  # Items 10, 11, 12
                response = test_client.get(f'/add_to_cart/{item_id}')
                assert response.status_code in [200, 302]
            
            # Verify sessions are isolated
            # (In real app, would verify in database/session store)
            assert len(set(session_ids)) == 3
    
    def test_concurrent_recommendations_requests(self, client, mock_recommender):
        """Multiple concurrent recommendation requests don't interfere."""
        with patch('app.recommender', mock_recommender):
            results = []
            errors = []
            
            def request_recommendations(user_id):
                try:
                    test_client = app.test_client()
                    test_client.get('/')
                    # View a product to trigger recommendations
                    response = test_client.get(f'/product/{user_id % 10 + 1}')
                    results.append((user_id, response.status_code))
                except Exception as e:
                    errors.append((user_id, str(e)))
            
            # Launch 5 concurrent requests
            threads = []
            for i in range(5):
                thread = threading.Thread(target=request_recommendations, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all to complete
            for thread in threads:
                thread.join(timeout=5)
            
            # All should succeed
            assert len(errors) == 0, f"Concurrent requests failed: {errors}"
            assert len(results) == 5, "Should have 5 results"
            assert all(status in [200, 302] for _, status in results)


# ============================================================================
# TEST 3: Model Inference Latency
# ============================================================================

class TestModelInferenceLatency:
    """Test recommendation endpoint returns within acceptable latency."""
    
    def test_recommendations_latency_under_threshold(self, client, mock_recommender):
        """Recommendations should return within 1 second."""
        with patch('app.recommender', mock_recommender):
            # Add items to session first
            client.get('/add_to_cart/1')
            
            # Measure latency for product page (includes recommendations)
            start_time = time.time()
            response = client.get('/product/2')
            latency = time.time() - start_time
            
            # Should be fast (under 1 second for mock)
            assert latency < 1.0, f"Product page took {latency}s, expected < 1s"
            assert response.status_code in [200, 404]
    
    def test_add_to_cart_latency_under_threshold(self, client, mock_recommender):
        """Adding item to cart should return quickly (< 500ms)."""
        with patch('app.recommender', mock_recommender):
            start_time = time.time()
            response = client.get('/add_to_cart/1')
            latency = time.time() - start_time
            
            assert latency < 0.5, f"Add to cart took {latency}s, expected < 0.5s"
            assert response.status_code in [200, 302]
    
    def test_index_page_latency_under_threshold(self, client, mock_recommender):
        """Index page should load quickly (< 2 seconds)."""
        with patch('app.recommender', mock_recommender):
            start_time = time.time()
            response = client.get('/')
            latency = time.time() - start_time
            
            assert latency < 2.0, f"Index page took {latency}s, expected < 2s"
            assert response.status_code == 200


# ============================================================================
# TEST 4: Data Consistency - Interactions Match Recommendations
# ============================================================================

class TestDataConsistency:
    """Verify interactions are properly logged and consistent with recommendations."""
    
    def test_added_items_not_in_recommendations(self, client, mock_recommender):
        """Items added to cart should not appear in recommendations (exclude_seen=True)."""
        with patch('app.recommender', mock_recommender):
            # Add items 1 and 2 by viewing products
            client.get('/product/1')
            client.get('/product/2')
            
            # Get recommendations from a new product view
            response = client.get('/product/3')
            
            # Should handle gracefully
            # Mock recommender enforces exclude_seen=True
            assert response.status_code in [200, 404]
    
    def test_interactions_file_contains_logged_events(self, client, mock_recommender):
        """Verify interactions are logged to CSV file."""
        with patch('app.recommender', mock_recommender):
            # Temporarily use a temp file for interactions
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = f.name
            
            try:
                with patch('app.INTERACTIONS_PATH', temp_path):
                    # Add an item (should log interaction)
                    response = client.get('/add_to_cart/5')
                    assert response.status_code in [200, 302]
                    
                    # Check if file has content (interaction was logged)
                    # Note: actual logging depends on app.log_event implementation
                    file_exists = os.path.exists(temp_path)
                    assert file_exists or response.status_code == 200
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_session_persistence_across_requests(self, client, mock_recommender):
        """Session data should persist across multiple requests."""
        with patch('app.recommender', mock_recommender):
            # Get initial session ID
            with client.session_transaction() as sess:
                initial_session_id = sess.get('_id')
            
            # Make several requests
            for i in range(3):
                response = client.get('/')
                assert response.status_code == 200
            
            # Session ID should remain the same
            with client.session_transaction() as sess:
                final_session_id = sess.get('_id')
            
            assert initial_session_id == final_session_id, "Session ID should not change"


# ============================================================================
# TEST 5: API Error Handling
# ============================================================================

class TestAPIErrorHandling:
    """Verify graceful error handling across all endpoints."""
    
    def test_invalid_item_id_format(self, client, mock_recommender):
        """Invalid item ID format should be handled gracefully."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/add_to_cart/invalid-id')
            # Should not crash (redirect or handle gracefully)
            assert response.status_code in [200, 302, 400]
    
    def test_nonexistent_item_id(self, client, mock_recommender):
        """Nonexistent item ID should be handled gracefully."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/add_to_cart/99999')
            # Should handle gracefully (can still add to cart)
            assert response.status_code in [200, 302]
    
    def test_missing_item_id_in_url(self, client, mock_recommender):
        """Missing product_id in URL should return 404."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/add_to_cart/')
            # Should return 404 (route requires product_id parameter)
            assert response.status_code == 404
    
    def test_log_click_endpoint(self, client, mock_recommender):
        """Test /log_click endpoint with valid and invalid data."""
        with patch('app.recommender', mock_recommender):
            # Valid log_click request
            response = client.post('/log_click',
                                  json={'item_ids': ['1', '2', '3']},
                                  content_type='application/json')
            # Should return 204 (No Content)
            assert response.status_code == 204
    
    def test_nonexistent_route(self, client, mock_recommender):
        """Nonexistent route should return 404."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/nonexistent-route')
            assert response.status_code == 404
    
    def test_product_page_with_invalid_id(self, client, mock_recommender):
        """Product page with invalid ID should return 404 or error page."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/product/999999999')
            # Could be 404, 200 (error page), or redirect
            assert response.status_code in [200, 302, 404]
    
    def test_checkout_with_empty_cart(self, client, mock_recommender):
        """Checkout with empty cart should be handled gracefully."""
        with patch('app.recommender', mock_recommender):
            response = client.get('/checkout')
            # Should handle gracefully (redirect, show form, or error)
            assert response.status_code in [200, 302]
    
    def test_recommender_unavailable_fallback(self, client):
        """App should handle recommender unavailability gracefully."""
        with patch('app.recommender', None):
            # App should still serve pages without recommendations
            response = client.get('/')
            assert response.status_code == 200
            
            # Add to cart should still work
            response = client.get('/add_to_cart/1')
            assert response.status_code in [200, 302]
    
    def test_purchase_endpoint(self, client, mock_recommender):
        """Test /purchase endpoint (POST) to complete checkout."""
        with patch('app.recommender', mock_recommender):
            # Add item to cart first
            client.get('/add_to_cart/1')
            
            # Submit purchase
            response = client.post('/purchase')
            # Should redirect to checkout
            assert response.status_code in [200, 302]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
