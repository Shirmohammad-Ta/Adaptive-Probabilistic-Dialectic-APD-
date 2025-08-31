#!/usr/bin/env python3
"""
Example usage of APD framework for novel idea generation.
"""

import os
import json
from typing import List, Dict
from src.apd_core import APDDialectic
from config_loader import ConfigLoader

class IdeaGenerationAPD(APDDialectic):
    """APD implementation for idea generation tasks"""
    
    def __init__(self, model_name: str = "gpt4o"):
        loader = ConfigLoader()
        config = loader.get_config(model_name)
        
        # Adjust parameters for idea generation
        super().__init__(
            model_name=config['model']['name'],
            tau_min=0.4,  # More exploration for ideas
            tau_max=0.9,
            tau_a=0.6,
            gamma=0.35,   # Higher novelty threshold
            window_size=2,
            p0=0.25,
            p1=0.65,
            beta=1.8,
            max_iter=5    # More iterations for creativity
        )
    
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
                max_tokens=500,
                timeout=45
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "Error: Unable to generate response"

def generate_initial_idea(topic: str, apd: IdeaGenerationAPD) -> str:
    """Generate initial idea on given topic"""
    prompt = f"""Generate an innovative idea about {topic}. 

Requirements:
1. Be creative and original
2. Provide clear explanation
3. Include potential applications
4. Keep it concise but comprehensive

Topic: {topic}
"""
    return apd._call_llm(prompt, temperature=0.8)

def run_idea_generation_demo():
    """Run demo for idea generation"""
    print("💡 APD Idea Generation Demo")
    print("=" * 60)
    
    # Initialize APD for idea generation
    apd = IdeaGenerationAPD(model_name="gpt4o")
    
    topics = [
        "renewable energy innovation",
        "AI in healthcare",
        "sustainable urban development",
        "quantum computing applications"
    ]
    
    results = []
    for i, topic in enumerate(topics, 1):
        print(f"\n🎯 Topic {i}: {topic}")
        
        # Generate initial idea
        initial_idea = generate_initial_idea(topic, apd)
        print("   Initial idea generated...")
        
        # Run APD process for idea refinement
        result = apd.run_dialectic(
            initial_question=f"Generate innovative ideas about {topic}",
            initial_solution=initial_idea
        )
        
        results.append({
            'topic': topic,
            'initial_idea': initial_idea,
            'final_idea': result['final_solution'],
            'iterations': result['total_iterations'],
            'avg_novelty': result['avg_novelty'],
            'history': result['history']
        })
        
        print(f"   Final idea obtained after {result['total_iterations']} iterations")
        print(f"   Average novelty: {result['avg_novelty']:.3f}")
        print(f"   Final idea preview: {result['final_solution'][:100]}...")
    
    # Save detailed results
    with open('idea_generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print creative summary
    print("\n" + "=" * 60)
    print("✨ Creativity Metrics:")
    total_iterations = sum(r['iterations'] for r in results)
    avg_novelty = sum(r['avg_novelty'] for r in results) / len(results)
    
    print(f"   Total dialectical iterations: {total_iterations}")
    print(f"   Overall novelty score: {avg_novelty:.3f}")
    print(f"   Topics explored: {len(topics)}")
    
    return results

def analyze_creativity(results: List[Dict]):
    """Analyze creativity metrics from results"""
    print("\n🔍 Creativity Analysis:")
    for result in results:
        novelty_scores = [step['novelty_score'] for step in result['history']]
        temperatures = [step['temperature'] for step in result['history']]
        
        print(f"\nTopic: {result['topic']}")
        print(f"  Novelty progression: {[f'{n:.2f}' for n in novelty_scores]}")
        print(f"  Temperature adaptation: {[f'{t:.2f}' for t in temperatures]}")
        print(f"  Creativity gain: {(result['avg_novelty'] - novelty_scores[0]):.3f}")

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    print("Starting idea generation demo...")
    results = run_idea_generation_demo()
    
    analyze_creativity(results)
    print("\n✅ Idea generation demo completed!")