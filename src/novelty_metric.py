import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class NoveltyMetric:
    """
    A class to compute semantic novelty between text segments using sentence embeddings.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu',
                 batch_size: int = 32):
        """
        Initialize the novelty metric calculator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for embedding computation
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name
        logger.info(f"Initialized NoveltyMetric with model: {model_name}")
    
    def compute_embeddings(self, 
                          texts: Union[str, List[str]]) -> np.ndarray:
        """
        Compute embeddings for input texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, 
                                     batch_size=self.batch_size,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
        return embeddings
    
    def calculate_novelty(self, 
                         text1: str, 
                         text2: str, 
                         metric: str = 'cosine') -> float:
        """
        Calculate novelty score between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Novelty score between 0 (identical) and 1 (completely different)
        """
        # Compute embeddings
        embeddings = self.compute_embeddings([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # Calculate similarity based on chosen metric
        if metric == 'cosine':
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            novelty = 1 - similarity
        elif metric == 'euclidean':
            distance = np.linalg.norm(emb1 - emb2)
            # Normalize to [0, 1] range (assuming embeddings are normalized)
            novelty = min(distance / 2.0, 1.0)  # Max Euclidean distance between normalized vectors is 2
        elif metric == 'manhattan':
            distance = np.sum(np.abs(emb1 - emb2))
            novelty = min(distance / 4.0, 1.0)  # Approximate normalization
        else:
            raise ValueError(f"Unsupported metric: {metric}. Choose from 'cosine', 'euclidean', 'manhattan'")
        
        logger.debug(f"Novelty calculation: similarity={1-novelty:.3f}, novelty={novelty:.3f}")
        return float(novelty)
    
    def batch_calculate_novelty(self, 
                               text_pairs: List[tuple], 
                               metric: str = 'cosine') -> List[float]:
        """
        Calculate novelty scores for multiple text pairs.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            metric: Similarity metric
            
        Returns:
            List of novelty scores
        """
        # Flatten texts for batch processing
        all_texts = []
        for text1, text2 in text_pairs:
            all_texts.extend([text1, text2])
        
        # Compute all embeddings
        all_embeddings = self.compute_embeddings(all_texts)
        
        # Calculate novelty for each pair
        novelty_scores = []
        for i in range(0, len(all_embeddings), 2):
            emb1, emb2 = all_embeddings[i], all_embeddings[i+1]
            
            if metric == 'cosine':
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                novelty = 1 - similarity
            elif metric == 'euclidean':
                distance = np.linalg.norm(emb1 - emb2)
                novelty = min(distance / 2.0, 1.0)
            elif metric == 'manhattan':
                distance = np.sum(np.abs(emb1 - emb2))
                novelty = min(distance / 4.0, 1.0)
            
            novelty_scores.append(float(novelty))
        
        return novelty_scores
    
    def is_novel(self, 
                text1: str, 
                text2: str, 
                threshold: float = 0.3,
                metric: str = 'cosine') -> bool:
        """
        Determine if text2 is novel compared to text1.
        
        Args:
            text1: Reference text
            text2: New text to evaluate
            threshold: Novelty threshold
            metric: Similarity metric
            
        Returns:
            True if novel, False otherwise
        """
        novelty = self.calculate_novelty(text1, text2, metric)
        return novelty > threshold

# Example usage
if __name__ == "__main__":
    # Initialize novelty metric
    novelty_calculator = NoveltyMetric()
    
    # Example texts
    text1 = "The cat sat on the mat."
    text2 = "The feline rested on the rug."
    text3 = "Dogs are playing in the park."
    
    # Calculate novelty
    novelty1 = novelty_calculator.calculate_novelty(text1, text2)
    novelty2 = novelty_calculator.calculate_novelty(text1, text3)
    
    print(f"Novelty between similar texts: {novelty1:.3f}")
    print(f"Novelty between different texts: {novelty2:.3f}")
    print(f"Is text2 novel compared to text1? {novelty_calculator.is_novel(text1, text2)}")
    print(f"Is text3 novel compared to text1? {novelty_calculator.is_novel(text1, text3)}")