#!/usr/bin/env python3
"""
Example of using APD with custom datasets.
"""

import os
import json
import pandas as pd
from typing import List, Dict
from src.apd_core import APDDialectic
from config_loader import ConfigLoader

class CustomDatasetAPD(APDDialectic):
    """APD implementation for custom datasets"""
    
    def __init__(self, model_name: str = "gpt4o", **kwargs):
        loader = ConfigLoader()
        config = loader.get_config(model_name)
        
        # Allow custom parameters or use defaults
        params = {
            'model_name': config['model']['name'],
            'tau_min': kwargs.get('tau_min', config['apd_parameters']['tau_min']),
            'tau_max': kwargs.get('tau_max', config['apd_parameters']['tau_max']),
            'tau_a': kwargs.get('tau_a', config['apd_parameters']['tau_a']),
            'gamma': kwargs.get('gamma', config['apd_parameters']['gamma']),
            'window_size': kwargs.get('window_size', config['apd_parameters']['novelty_window_size']),
            'p0': kwargs.get('p0', config['apd_parameters']['sprt']['p0']),
            'p1': kwargs.get('p1', config['apd_parameters']['sprt']['p1']),
            'beta': kwargs.get('beta', config['apd_parameters']['beta']),
            'max_iter': kwargs.get('max_iter', config['apd_parameters']['max_iterations'])
        }
        
        super().__init__(**params)
        self.config = config
    
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """Implementation using OpenAI API"""
        import openai
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.config['model']['max_tokens'],
                timeout=self.config['model']['timeout']
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "Error: Unable to generate response"

def load_custom_dataset(file_path: str, format_type: str = 'json') -> List[Dict]:
    """Load custom dataset from file"""
    if format_type == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif format_type == 'csv':
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def create_custom_prompt(problem: Dict, prompt_template: str) -> str:
    """Create prompt from custom problem"""
    # Customize this based on your dataset structure
    if 'question' in problem and 'context' in problem:
        return prompt_template.format(
            context=problem['context'],
            question=problem['question']
        )
    elif 'problem' in problem:
        return prompt_template.format(problem=problem['problem'])
    else:
        return prompt_template.format(**problem)

def run_custom_dataset_demo(dataset_path: str, dataset_format: str = 'json'):
    """Run demo with custom dataset"""
    print("📁 Custom Dataset APD Demo")
    print("=" * 50)
    
    # Load custom dataset
    try:
        dataset = load_custom_dataset(dataset_path, dataset_format)
        print(f"Loaded {len(dataset)} problems from {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize APD with custom parameters
    apd = CustomDatasetAPD(
        model_name="gpt4o",
        max_iter=4,      # Custom iteration limit
        gamma=0.4,       # Custom novelty threshold
        tau_max=0.95     # More exploration
    )
    
    results = []
    for i, problem in enumerate(dataset[:3], 1):  # Limit to 3 for demo
        print(f"\n📊 Processing problem {i}/{min(3, len(dataset))}")
        
        # Create initial prompt (customize based on your dataset)
        initial_prompt = create_custom_prompt(
            problem, 
            "Solve the following problem: {problem}\n\nProvide step-by-step reasoning."
        )
        
        # Generate initial solution
        initial_solution = apd._call_llm(initial_prompt, temperature=0.7)
        
        # Run APD process
        result = apd.run_dialectic(
            initial_question=str(problem),
            initial_solution=initial_solution
        )
        
        results.append({
            'problem': problem,
            'initial_solution': initial_solution,
            'final_solution': result['final_solution'],
            'iterations': result['total_iterations'],
            'avg_novelty': result['avg_novelty'],
            'history': result['history']
        })
        
        print(f"   Completed {result['total_iterations']} iterations")
        print(f"   Novelty score: {result['avg_novelty']:.3f}")
    
    # Save results
    output_file = f"custom_dataset_results_{os.path.basename(dataset_path).split('.')[0]}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Demo completed! Results saved to {output_file}")
    return results

def create_sample_dataset():
    """Create sample custom dataset for testing"""
    sample_data = [
        {
            "problem": "If a train travels at 80 km/h for 2 hours, how far does it go?",
            "type": "physics",
            "difficulty": "easy"
        },
        {
            "problem": "Calculate the area of a circle with radius 5 units.",
            "type": "math",
            "difficulty": "medium"
        },
        {
            "context": "In a basketball game, Team A scored 85 points and Team B scored 78 points.",
            "question": "What was the point difference?",
            "type": "sports",
            "difficulty": "easy"
        }
    ]
    
    with open('sample_dataset.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created sample_dataset.json for testing")

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    # Create sample dataset if needed
    if not os.path.exists('sample_dataset.json'):
        create_sample_dataset()
    
    # Run demo with sample dataset
    results = run_custom_dataset_demo('sample_dataset.json', 'json')
    
    # Example of custom parameter tuning
    print("\n🎛️  Example of parameter tuning:")
    tuned_apd = CustomDatasetAPD(
        model_name="gpt4o-mini",
        tau_min=0.5,
        tau_max=1.0,    # Maximum creativity
        gamma=0.2,      # Lower novelty threshold
        max_iter=5      # More iterations
    )
    print("Tuned APD instance created for specific use case")