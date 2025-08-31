#!/usr/bin/env python3
"""
Unit tests for SPRTOptimizer class.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sprt_optimizer import SPRTOptimizer, SPRTConfig

class TestSPRTOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SPRTConfig(p0=0.2, p1=0.6, alpha=0.05, beta=0.05)
        self.sprt = SPRTOptimizer(self.config)
    
    def test_initialization(self):
        """Test that SPRTOptimizer initializes correctly."""
        self.assertEqual(self.sprt.config.p0, 0.2)
        self.assertEqual(self.sprt.config.p1, 0.6)
        self.assertEqual(self.sprt.config.alpha, 0.05)
        self.assertEqual(self.sprt.config.beta, 0.05)
        
        # Test threshold calculations
        expected_A = np.log((1 - 0.05) / 0.05)  # (1-β)/α
        expected_B = np.log(0.05 / (1 - 0.05))  # β/(1-α)
        
        self.assertAlmostEqual(self.sprt.A, expected_A, places=5)
        self.assertAlmostEqual(self.sprt.B, expected_B, places=5)
        self.assertEqual(self.sprt.llr, 0.0)
    
    def test_update_novel_observation(self):
        """Test update with novel observation (True)."""
        decision = self.sprt.update(True)
        
        # LLR should increase for novel observation
        expected_llr = np.log(0.6 / 0.2)  # log(p1/p0)
        self.assertAlmostEqual(self.sprt.llr, expected_llr, places=5)
        self.assertEqual(decision, 'continue')  # Should continue initially
    
    def test_update_non_novel_observation(self):
        """Test update with non-novel observation (False)."""
        decision = self.sprt.update(False)
        
        # LLR should decrease for non-novel observation
        expected_llr = np.log((1 - 0.6) / (1 - 0.2))  # log((1-p1)/(1-p0))
        self.assertAlmostEqual(self.sprt.llr, expected_llr, places=5)
        self.assertEqual(decision, 'continue')
    
    def test_decision_boundaries_novel(self):
        """Test that SPRT stops when LLR exceeds upper threshold."""
        # Force LLR above upper threshold
        self.sprt.llr = self.sprt.A + 1.0
        decision = self.sprt.update(True)  # Any observation
        
        self.assertEqual(decision, 'stop')
    
    def test_decision_boundaries_non_novel(self):
        """Test that SPRT stops when LLR falls below lower threshold."""
        # Force LLR below lower threshold
        self.sprt.llr = self.sprt.B - 1.0
        decision = self.sprt.update(False)  # Any observation
        
        self.assertEqual(decision, 'stop')
    
    def test_decision_continuation(self):
        """Test that SPRT continues when LLR is between thresholds."""
        # Set LLR to middle value
        self.sprt.llr = (self.sprt.A + self.sprt.B) / 2
        decision = self.sprt.update(True)
        
        self.assertEqual(decision, 'continue')
    
    def test_get_decision_boundaries(self):
        """Test get_decision_boundaries method."""
        boundaries = self.sprt.get_decision_boundaries()
        
        self.assertEqual(boundaries['upper_threshold'], self.sprt.A)
        self.assertEqual(boundaries['lower_threshold'], self.sprt.B)
        self.assertEqual(boundaries['current_llr'], 0.0)
        
        # Test distances
        self.assertEqual(boundaries['distance_to_upper'], self.sprt.A)
        self.assertEqual(boundaries['distance_to_lower'], -self.sprt.B)
    
    def test_get_sample_size_estimate(self):
        """Test sample size estimation."""
        sample_info = self.sprt.get_sample_size_estimate()
        
        self.assertIn('expected_sample_size_H0', sample_info)
        self.assertIn('expected_sample_size_H1', sample_info)
        self.assertIn('current_sample_size', sample_info)
        
        # Current sample size should be 0 initially
        self.assertEqual(sample_info['current_sample_size'], 0)
    
    def test_reset(self):
        """Test reset functionality."""
        # Make some updates
        self.sprt.update(True)
        self.sprt.update(False)
        
        # Reset
        self.sprt.reset()
        
        # Should be back to initial state
        self.assertEqual(self.sprt.llr, 0.0)
        self.assertEqual(len(self.sprt.observations), 0)
        self.assertEqual(len(self.sprt.decisions), 0)
    
    def test_get_history(self):
        """Test history retrieval."""
        # Make some updates
        self.sprt.update(True)
        self.sprt.update(False)
        
        history = self.sprt.get_history()
        
        self.assertEqual(history['observations'], [True, False])
        self.assertEqual(len(history['decisions']), 2)
        self.assertEqual(len(history['llr_history']), 2)
        self.assertEqual(history['config']['p0'], 0.2)
    
    def test_edge_case_p0_0(self):
        """Test edge case where p0 = 0."""
        with self.assertRaises(ValueError):
            config = SPRTConfig(p0=0.0, p1=0.6, alpha=0.05, beta=0.05)
            sprt = SPRTOptimizer(config)
            sprt.get_sample_size_estimate()
    
    def test_edge_case_p1_1(self):
        """Test edge case where p1 = 1."""
        with self.assertRaises(ValueError):
            config = SPRTConfig(p0=0.2, p1=1.0, alpha=0.05, beta=0.05)
            sprt = SPRTOptimizer(config)
            sprt.get_sample_size_estimate()

class TestSPRTConfig(unittest.TestCase):
    
    def test_config_default_values(self):
        """Test SPRTConfig default values."""
        config = SPRTConfig()
        
        self.assertEqual(config.p0, 0.2)
        self.assertEqual(config.p1, 0.6)
        self.assertEqual(config.alpha, 0.05)
        self.assertEqual(config.beta, 0.05)
    
    def test_config_custom_values(self):
        """Test SPRTConfig with custom values."""
        config = SPRTConfig(p0=0.1, p1=0.8, alpha=0.01, beta=0.01)
        
        self.assertEqual(config.p0, 0.1)
        self.assertEqual(config.p1, 0.8)
        self.assertEqual(config.alpha, 0.01)
        self.assertEqual(config.beta, 0.01)

if __name__ == '__main__':
    unittest.main()