from src.apd_core import APDDialectic
import openai

class OpenAIAPD(APDDialectic):
    def _call_llm(self, prompt: str, temperature: float) -> str:
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

# Initialize APD with GPT-4o
apd = OpenAIAPD(model_name="gpt-4o")

# Run on GSM8K problem
result = apd.run_dialectic(
    initial_question="John has 5 apples. He gives 2 to Mary and buys 3 more. How many does he have?",
    initial_solution="5 - 2 = 3, so he has 3 apples."
)

print(f"Final solution: {result['final_solution']}")
print(f"Total iterations: {result['total_iterations']}")
print(f"Average novelty: {result['avg_novelty']:.3f}")