import numpy as np
from typing import Optional, Literal
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SPRTConfig:
    """Configuration for Sequential Probability Ratio Test."""
    p0: float = 0.2  # Null hypothesis: low novelty rate
    p1: float = 0.6  # Alternative hypothesis: high novelty rate
    alpha: float = 0.05  # Type I error probability
    beta: float = 0.05  # Type II error probability

class SPRTOptimizer:
    """
    Sequential Probability Ratio Test for optimal stopping decisions.
    """
    
    def __init__(self, config: Optional[SPRTConfig] = None):
        """
        Initialize SPRT optimizer.
        
        Args:
            config: SPRT configuration parameters
        """
        self.config = config or SPRTConfig()
        self.llr: float = 0.0  # Log-likelihood ratio
        self.observations: List[bool] = []  # History of observations
        self.decisions: List[str] = []  # History of decisions
        
        # Calculate thresholds
        self.A = np.log((1 - self.config.beta) / self.config.alpha)  # Upper threshold (reject H0)
        self.B = np.log(self.config.beta / (1 - self.config.alpha))  # Lower threshold (accept H0)
        
        logger.info("Initialized SPRTOptimizer with config: "
                   f"p0={self.config.p0}, p1={self.config.p1}, "
                   f"alpha={self.config.alpha}, beta={self.config.beta}")
        logger.info(f"SPRT thresholds: A={self.A:.3f} (reject H0), B={self.B:.3f} (accept H0)")
    
    def update(self, observation: bool) -> Literal['continue', 'stop', 'uncertain']:
        """
        Update SPRT with new observation and return decision.
        
        Args:
            observation: Boolean indicating novelty (True) or no novelty (False)
            
        Returns:
            Decision: 'continue', 'stop', or 'uncertain'
        """
        self.observations.append(observation)
        
        # Update log-likelihood ratio
        if observation:
            self.llr += np.log(self.config.p1 / self.config.p0)
        else:
            self.llr += np.log((1 - self.config.p1) / (1 - self.config.p0))
        
        # Make decision
        if self.llr >= self.A:
            decision = 'stop'  # Reject H0 (sufficient novelty)
            logger.info(f"SPRT decision: STOP (LLR={self.llr:.3f} >= A={self.A:.3f})")
        elif self.llr <= self.B:
            decision = 'stop'  # Accept H0 (insufficient novelty)
            logger.info(f"SPRT decision: STOP (LLR={self.llr:.3f} <= B={self.B:.3f})")
        else:
            decision = 'continue'  # Continue sampling
            logger.debug(f"SPRT decision: CONTINUE (LLR={self.llr:.3f})")
        
        self.decisions.append(decision)
        return decision
    
    def get_decision_boundaries(self) -> dict:
        """
        Get current decision boundaries.
        
        Returns:
            Dictionary with current boundaries and LLR
        """
        return {
            'upper_threshold': self.A,
            'lower_threshold': self.B,
            'current_llr': self.llr,
            'distance_to_upper': self.A - self.llr,
            'distance_to_lower': self.llr - self.B
        }
    
    def get_sample_size_estimate(self) -> dict:
        """
        Estimate required sample size under both hypotheses.
        
        Returns:
            Dictionary with expected sample size estimates
        """
        # Expected sample size under H0
        if self.config.p0 == 0 or self.config.p0 == 1:
            E_N_H0 = float('inf')
        else:
            E_N_H0 = (self.config.alpha * np.log(self.config.beta/(1-self.config.alpha)) + 
                     (1-self.config.alpha) * np.log((1-self.config.beta)/self.config.alpha))
            E_N_H0 /= (self.config.p0 * np.log(self.config.p1/self.config.p0) + 
                      (1-self.config.p0) * np.log((1-self.config.p1)/(1-self.config.p0)))
        
        # Expected sample size under H1
        if self.config.p1 == 0 or self.config.p1 == 1:
            E_N_H1 = float('inf')
        else:
            E_N_H1 = (self.config.beta * np.log(self.config.beta/(1-self.config.alpha)) + 
                     (1-self.config.beta) * np.log((1-self.config.beta)/self.config.alpha))
            E_N_H1 /= (self.config.p1 * np.log(self.config.p1/self.config.p0) + 
                      (1-self.config.p1) * np.log((1-self.config.p1)/(1-self.config.p0)))
        
        return {
            'expected_sample_size_H0': abs(E_N_H0),
            'expected_sample_size_H1': abs(E_N_H1),
            'current_sample_size': len(self.observations)
        }
    
    def reset(self):
        """Reset SPRT state."""
        self.llr = 0.0
        self.observations = []
        self.decisions = []
        logger.info("SPRTOptimizer reset")
    
    def get_history(self) -> dict:
        """
        Get SPRT history.
        
        Returns:
            Dictionary with observation and decision history
        """
        return {
            'observations': self.observations.copy(),
            'decisions': self.decisions.copy(),
            'llr_history': [self.llr] * len(self.observations),  # Cumulative LLR
            'config': self.config.__dict__
        }

# Example usage
if __name__ == "__main__":
    # Initialize SPRT
    config = SPRTConfig(p0=0.2, p1=0.6, alpha=0.05, beta=0.05)
    sprt = SPRTOptimizer(config)
    
    # Simulate observations (True = novel, False = not novel)
    observations = [True, True, False, True, False, False, False]
    
    print("SPRT simulation:")
    for i, obs in enumerate(observations):
        decision = sprt.update(obs)
        boundaries = sprt.get_decision_boundaries()
        print(f"Observation {i+1}: novel={obs}, LLR={boundaries['current_llr']:.3f}, "
              f"Decision={decision}")
        
        if decision == 'stop':
            print("Stopping early due to SPRT decision")
            break
    
    # Get sample size estimates
    sample_info = sprt.get_sample_size_estimate()
    print(f"\nExpected sample size under H0: {sample_info['expected_sample_size_H0']:.2f}")
    print(f"Expected sample size under H1: {sample_info['expected_sample_size_H1']:.2f}")
    print(f"Actual sample size: {sample_info['current_sample_size']}")