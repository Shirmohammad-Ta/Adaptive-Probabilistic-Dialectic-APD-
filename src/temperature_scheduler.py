import numpy as np
from typing import List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TemperatureConfig:
    """Configuration for temperature scheduling."""
    tau_min: float = 0.3
    tau_max: float = 0.9
    beta: float = 2.0
    window_size: int = 2
    decay_type: str = 'exponential'  # 'exponential', 'linear', 'step'

class TemperatureScheduler:
    """
    Dynamic temperature scheduler for adaptive exploration-exploitation trade-off.
    """
    
    def __init__(self, config: Optional[TemperatureConfig] = None):
        """
        Initialize temperature scheduler.
        
        Args:
            config: Temperature configuration parameters
        """
        self.config = config or TemperatureConfig()
        self.novelty_history: List[float] = []
        self.temperature_history: List[float] = []
        
        logger.info("Initialized TemperatureScheduler with config: "
                   f"tau_min={self.config.tau_min}, tau_max={self.config.tau_max}, "
                   f"beta={self.config.beta}, window_size={self.config.window_size}")
    
    def update_novelty_history(self, novelty_score: float):
        """
        Update novelty history with latest score.
        
        Args:
            novelty_score: Latest novelty score (0-1)
        """
        self.novelty_history.append(novelty_score)
        # Keep only the most recent scores based on window size
        if len(self.novelty_history) > self.config.window_size:
            self.novelty_history = self.novelty_history[-self.config.window_size:]
    
    def get_moving_average_novelty(self) -> float:
        """
        Calculate moving average of recent novelty scores.
        
        Returns:
            Moving average novelty score
        """
        if not self.novelty_history:
            return 0.5  # Default value when no history
        
        return float(np.mean(self.novelty_history))
    
    def calculate_temperature(self, 
                            iteration: Optional[int] = None,
                            current_novelty: Optional[float] = None) -> float:
        """
        Calculate dynamic temperature based on novelty history.
        
        Args:
            iteration: Current iteration number (for step decay)
            current_novelty: Current novelty score (optional)
            
        Returns:
            Dynamic temperature value
        """
        if current_novelty is not None:
            self.update_novelty_history(current_novelty)
        
        P_hat = self.get_moving_average_novelty()
        
        if self.config.decay_type == 'exponential':
            # Exponential decay based on novelty
            tau = self.config.tau_min + (self.config.tau_max - self.config.tau_min) * np.exp(-self.config.beta * P_hat)
        
        elif self.config.decay_type == 'linear':
            # Linear decay based on novelty
            tau = self.config.tau_max - (self.config.tau_max - self.config.tau_min) * P_hat
        
        elif self.config.decay_type == 'step':
            # Step decay based on iteration
            if iteration is None:
                raise ValueError("Iteration required for step decay")
            decay_steps = 3  # Number of steps to decay
            if iteration < decay_steps:
                tau = self.config.tau_max - (self.config.tau_max - self.config.tau_min) * (iteration / decay_steps)
            else:
                tau = self.config.tau_min
        
        else:
            raise ValueError(f"Unknown decay type: {self.config.decay_type}")
        
        # Clip to valid range
        tau = np.clip(tau, self.config.tau_min, self.config.tau_max)
        self.temperature_history.append(tau)
        
        logger.debug(f"Temperature calculation: P_hat={P_hat:.3f}, tau={tau:.3f}")
        return tau
    
    def reset(self):
        """Reset scheduler state."""
        self.novelty_history = []
        self.temperature_history = []
        logger.info("TemperatureScheduler reset")
    
    def get_history(self) -> dict:
        """
        Get scheduling history.
        
        Returns:
            Dictionary with novelty and temperature history
        """
        return {
            'novelty_history': self.novelty_history.copy(),
            'temperature_history': self.temperature_history.copy(),
            'config': self.config.__dict__
        }

# Example usage
if __name__ == "__main__":
    # Initialize scheduler
    config = TemperatureConfig(tau_min=0.3, tau_max=0.9, beta=2.0, window_size=2)
    scheduler = TemperatureScheduler(config)
    
    # Simulate novelty scores over iterations
    novelty_scores = [0.8, 0.6, 0.4, 0.2, 0.1]
    
    print("Temperature scheduling simulation:")
    for i, novelty in enumerate(novelty_scores):
        temperature = scheduler.calculate_temperature(iteration=i, current_novelty=novelty)
        print(f"Iteration {i+1}: novelty={novelty:.2f}, temperature={temperature:.2f}")