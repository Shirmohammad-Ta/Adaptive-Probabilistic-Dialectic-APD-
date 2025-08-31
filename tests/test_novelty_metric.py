#!/usr/bin/env python3
"""
Unit tests for NoveltyMetric class.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from novelty_metric import NoveltyMetric

class TestNoveltyMetric(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the SentenceTransformer to avoid loading real model
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_model.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            self.novelty_metric = NoveltyMetric(model_name='all-MiniLM-L6-v2')
    
    def test_initialization(self):
        """Test that NoveltyMetric initializes correctly."""
        self.assertIsNotNone(self.novelty_metric)
        self.assertIsNotNone(self.novelty_metric.model)
        self.assertEqual(self.novelty_metric.batch_size, 32)
    
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_compute_embeddings_single_text(self, mock_encode):
        """Test embedding computation for single text."""
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        embeddings = self.novelty_metric.compute_embeddings("Hello world")
        
        self.assertEqual(embeddings.shape, (1, 3))
        mock_encode.assert_called_once_with(["Hello world"], batch_size=32, 
                                          convert_to_numpy=True, normalize_embeddings=True)
    
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_compute_embeddings_multiple_texts(self, mock_encode):
        """Test embedding computation for multiple texts."""
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        texts = ["Hello", "World"]
        embeddings = self.novelty_metric.compute_embeddings(texts)
        
        self.assertEqual(embeddings.shape, (2, 3))
        mock_encode.assert_called_once_with(texts, batch_size=32, 
                                          convert_to_numpy=True, normalize_embeddings=True)
    
    @patch('novelty_metric.NoveltyMetric.compute_embeddings')
    def test_calculate_novelty_cosine(self, mock_compute_embeddings):
        """Test cosine novelty calculation."""
        # Mock embeddings for two texts
        mock_compute_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],  # Text 1
            [0.0, 1.0, 0.0]   # Text 2 (orthogonal to text 1)
        ])
        
        novelty = self.novelty_metric.calculate_novelty("Text 1", "Text 2", metric='cosine')
        
        # Cosine similarity between orthogonal vectors is 0, so novelty should be 1
        self.assertAlmostEqual(novelty, 1.0, places=5)
    
    @patch('novelty_metric.NoveltyMetric.compute_embeddings')
    def test_calculate_novelty_identical_texts(self, mock_compute_embeddings):
        """Test novelty calculation for identical texts."""
        mock_compute_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]  # Same embedding
        ])
        
        novelty = self.novelty_metric.calculate_novelty("Same", "Same", metric='cosine')
        
        # Identical texts should have novelty 0
        self.assertAlmostEqual(novelty, 0.0, places=5)
    
    @patch('novelty_metric.NoveltyMetric.compute_embeddings')
    def test_calculate_novelty_euclidean(self, mock_compute_embeddings):
        """Test Euclidean novelty calculation."""
        mock_compute_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]  # Distance = sqrt(2)
        ])
        
        novelty = self.novelty_metric.calculate_novelty("Text 1", "Text 2", metric='euclidean')
        
        # Normalized Euclidean distance for unit vectors
        expected_novelty = min(np.sqrt(2) / 2.0, 1.0)
        self.assertAlmostEqual(novelty, expected_novelty, places=5)
    
    def test_calculate_novelty_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with self.assertRaises(ValueError):
            self.novelty_metric.calculate_novelty("Text 1", "Text 2", metric='invalid_metric')
    
    @patch('novelty_metric.NoveltyMetric.compute_embeddings')
    def test_is_novel(self, mock_compute_embeddings):
        """Test is_novel method with threshold."""
        mock_compute_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]  # Orthogonal - high novelty
        ])
        
        # High novelty should be above threshold
        is_novel = self.novelty_metric.is_novel("Text 1", "Text 2", threshold=0.5)
        self.assertTrue(is_novel)
        
        # Test with very high threshold
        is_novel = self.novelty_metric.is_novel("Text 1", "Text 2", threshold=0.95)
        self.assertFalse(is_novel)
    
    @patch('novelty_metric.NoveltyMetric.compute_embeddings')
    def test_batch_calculate_novelty(self, mock_compute_embeddings):
        """Test batch novelty calculation."""
        # Mock embeddings for 2 text pairs (4 texts total)
        mock_compute_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],  # Pair 1, Text 1
            [0.0, 1.0, 0.0],  # Pair 1, Text 2 (orthogonal)
            [1.0, 0.0, 0.0],  # Pair 2, Text 1
            [1.0, 0.0, 0.0]   # Pair 2, Text 2 (identical)
        ])
        
        text_pairs = [("A1", "A2"), ("B1", "B2")]
        novelty_scores = self.novelty_metric.batch_calculate_novelty(text_pairs, metric='cosine')
        
        self.assertEqual(len(novelty_scores), 2)
        # First pair: orthogonal → novelty ~1.0
        self.assertAlmostEqual(novelty_scores[0], 1.0, places=5)
        # Second pair: identical → novelty 0.0
        self.assertAlmostEqual(novelty_scores[1], 0.0, places=5)

if __name__ == '__main__':
    unittest.main()