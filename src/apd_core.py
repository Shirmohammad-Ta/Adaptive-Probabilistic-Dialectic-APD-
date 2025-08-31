import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from scipy import stats

class APDDialectic:
    def __init__(self, 
                 model_name: str,
                 tau_min: float = 0.3,
                 tau_max: float = 0.9,
                 tau_a: float = 0.5,
                 gamma: float = 0.3,
                 window_size: int = 2,
                 p0: float = 0.2,
                 p1: float = 0.6,
                 beta: float = 2.0,
                 max_iter: int = 3):
        
        self.model_name = model_name
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_a = tau_a
        self.gamma = gamma
        self.window_size = window_size
        self.p0 = p0
        self.p1 = p1
        self.beta = beta
        self.max_iter = max_iter
        
        # Initialize components
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.novelty_history = []
        self.temperature_history = []
        self.llr = 0.0  # Log-likelihood ratio for SPRT
        
        # SPRT thresholds
        self.A = np.log((1 - p1) / (1 - p0))  # Upper threshold (reject H0)
        self.B = np.log(p1 / p0)              # Lower threshold (accept H0)
    
    def calculate_novelty(self, text1: str, text2: str) -> float:
        """Calculate semantic novelty between two texts"""
        emb1 = self.embedder.encode([text1])[0]
        emb2 = self.embedder.encode([text2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - similarity
    
    def get_dynamic_temperature(self) -> float:
        """Compute adaptive temperature based on recent novelty"""
        if len(self.novelty_history) < self.window_size:
            return self.tau_max
        
        recent_novelty = np.mean(self.novelty_history[-self.window_size:])
        tau = self.tau_min + (self.tau_max - self.tau_min) * np.exp(-self.beta * recent_novelty)
        return np.clip(tau, self.tau_min, self.tau_max)
    
    def update_sprt(self, is_novel: bool) -> str:
        """Update SPRT and return decision"""
        if is_novel:
            self.llr += np.log(self.p1 / self.p0)
        else:
            self.llr += np.log((1 - self.p1) / (1 - self.p0))
        
        if self.llr >= self.A:
            return 'continue'
        elif self.llr <= self.B:
            return 'stop'
        else:
            return 'uncertain'
    
    def generate_opposition(self, proposition: str, temperature: float) -> str:
        """Generate opposition using LLM"""
        # Implementation depends on LLM API (OpenAI, Anthropic, etc.)
        prompt = self._load_prompt('opposition_prompt.txt').format(
            proposition=proposition
        )
        return self._call_llm(prompt, temperature)
    
    def generate_unification(self, proposition: str, opposition: str, temperature: float) -> str:
        """Generate unified solution"""
        prompt = self._load_prompt('unification_prompt.txt').format(
            proposition=proposition,
            opposition=opposition
        )
        return self._call_llm(prompt, temperature)
    
    def run_dialectic(self, initial_question: str, initial_solution: str) -> Dict[str, Any]:
        """Execute full APD process"""
        current_solution = initial_solution
        history = []
        
        for iteration in range(self.max_iter):
            # Generate opposition
            opposition = self.generate_opposition(current_solution, self.tau_a)
            
            # Calculate dynamic temperature
            current_tau = self.get_dynamic_temperature()
            self.temperature_history.append(current_tau)
            
            # Generate unified solution
            unified_solution = self.generate_unification(
                current_solution, opposition, current_tau
            )
            
            # Calculate novelty
            novelty_score = self.calculate_novelty(unified_solution, current_solution)
            self.novelty_history.append(novelty_score)
            
            # Update SPRT
            is_novel = novelty_score > self.gamma
            decision = self.update_sprt(is_novel)
            
            # Record iteration history
            history.append({
                'iteration': iteration + 1,
                'proposition': current_solution,
                'opposition': opposition,
                'unified_solution': unified_solution,
                'novelty_score': novelty_score,
                'temperature': current_tau,
                'sprt_decision': decision,
                'sprt_llr': self.llr
            })
            
            # Check stopping condition
            if decision == 'stop':
                break
                
            current_solution = unified_solution
        
        return {
            'final_solution': current_solution,
            'history': history,
            'total_iterations': len(history),
            'avg_novelty': np.mean(self.novelty_history) if self.novelty_history else 0,
            'avg_temperature': np.mean(self.temperature_history) if self.temperature_history else 0
        }
    
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """Abstract method for LLM API call"""
        # To be implemented based on specific LLM provider
        raise NotImplementedError("Subclasses must implement _call_llm method")
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file"""
        with open(f'prompts/{filename}', 'r') as f:
            return f.read().strip()