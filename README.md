
# 🔁 Adaptive Probabilistic Dialectic (APD) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A novel framework for enhancing Large Language Model reasoning through Hegelian dialectical processes combined with adaptive probabilistic methods.

---

## 🎯 Overview

APD implements a self-reflective dialectical process inspired by Hegelian philosophy (Thesis → Antithesis → Synthesis) with modern machine learning techniques. The framework enables LLMs to:

- **Self-criticize** and identify errors in their own reasoning  
- **Generate opposing perspectives** to challenge initial solutions  
- **Synthesize improved solutions** through adaptive temperature scheduling  
- **Optimize computation** via Sequential Probability Ratio Test (SPRT) for early stopping  

---

## ✨ Key Features

- **🤖 Multi-Model Support**: GPT-4o, GPT-4o-mini, GPT-4-32k with optimized configurations  
- **🎛️ Adaptive Temperature Control**: Dynamic exploration-exploitation balance based on novelty metrics  
- **📊 Objective Novelty Assessment**: Semantic similarity-based novelty measurement  
- **⏱️ Optimal Stopping**: SPRT for efficient computation allocation  
- **📈 Comprehensive Evaluation**: Quantitative and qualitative analysis tools  
- **🔧 Modular Architecture**: Easily extensible components  

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-username/APD.git
cd APD

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY='your-api-key-here'
```

---



🚀 Quick Start

## Basic Usage

```python
from src.apd_core import APDDialectic
from config_loader import ConfigLoader

# Initialize with model-specific configuration
loader = ConfigLoader()
config = loader.get_config("gpt4o")

apd = APDDialectic(
    model_name=config['model']['name'],
    tau_min=config['apd_parameters']['tau_min'],
    tau_max=config['apd_parameters']['tau_max'],
    gamma=config['apd_parameters']['gamma']
)

# Run dialectical process
result = apd.run_dialectic(
    initial_question="What is 15% of 200?",
    initial_solution="15% of 200 is 30."
)

print(f"Final solution: {result['final_solution']}")
print(f"Iterations: {result['total_iterations']}")
```

## Example Runs

* Mathematical Reasoning:

```bash
python examples/gsm8k_example.py
```

* Creative Idea Generation:

```bash
python examples/idea_generation_example.py
```

* Custom Datasets:

```bash
python examples/custom_dataset_example.py
```

🏗️ **Architecture**

```
APD/
├── src/
│   ├── apd_core.py          # Main dialectical algorithm
│   ├── novelty_metric.py    # Semantic novelty calculation
│   ├── temperature_scheduler.py # Adaptive temperature control
│   └── sprt_optimizer.py    # Optimal stopping with SPRT
├── configs/
│   ├── gpt4o_config.yaml    # Model-specific configurations
│   ├── gpt4o_mini_config.yaml
│   └── gpt4_32k_config.yaml
├── examples/                # Usage examples
├── prompts/                 # Prompt templates
└── results/                 # Output analysis and visualization
```

⚙️ **Configuration**

* APD uses YAML configuration files for model-specific tuning.

Example:

```yaml
# configs/gpt4o_config.yaml
model:
  name: "gpt-4o"
  max_tokens: 4096
  timeout: 30

apd_parameters:
  tau_min: 0.3
  tau_max: 0.9
  gamma: 0.3
  max_iterations: 3
```

Available configurations:

* `gpt4o_config.yaml`: Optimized for GPT-4o
* `gpt4o_mini_config.yaml`: Tuned for GPT-4o-mini
* `gpt4_32k_config.yaml`: Configured for GPT-4-32k

📊 **Performance**

* GSM8K Mathematical Reasoning

| Model       | Baseline | APD  | Improvement |
|------------|---------|------|-------------|
| GPT-4o     | 95.1%   | 96.3% | +1.26%      |
| GPT-4o-mini| 93.7%   | 95.2% | +1.60%      |
| GPT-4-32k  | 95.2%   | 96.6% | +1.47%      |


Efficiency Metrics:

* 28% reduction in unnecessary iterations
* 22% faster convergence compared to fixed iteration approaches
* Adaptive computation based on problem complexity

🎨 **Customization**

* Custom Prompts: Modify prompt templates in `prompts/` directory.
* Novelty Metrics: Implement custom novelty measures using `novelty_metric.py`

📈 **Results Analysis**

* Results are stored in structured format under `results/` directory.
* Analyze results with provided tools:

```bash
python analyze_results.py results/gsm8k/2024-01-15/results.json
python generate_plots.py results/gsm8k/2024-01-15/results.json
```

🔬 **Research Basis**

* Hegelian Dialectics (Thesis-Antithesis-Synthesis)
* Sequential Probability Ratio Test (Optimal stopping)
* Simulated Annealing (Temperature scheduling)
* Semantic Similarity (Novelty measurement)

🚧 **Limitations and Future Work**

* Dependency on external LLM APIs
* Limited to text-based reasoning tasks
* Requires careful prompt engineering

Planned Enhancements:

* Support for local LLM models
* Multi-modal capabilities
* Automated prompt optimization
* Real-time adaptation learning

🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open a Pull Request

📝 **Citation**

```bibtex
@software{apd2024,
  title = {Adaptive Probabilistic Dialectic Framework},
  author = {Your Name and Collaborators},
  year = {2024},
  url = {https://github.com/your-username/APD}
}
```

📄 **License**

* MIT License (see LICENSE file)

🙏 **Acknowledgments**

* Inspired by Hegelian philosophical principles
* Built upon advancements in LLM self-reflection research
* Thanks to the open-source community

## 📞 Contact
- **Author:** Shirmohammad Tavangari  
- **Email:** s.tavangari@alumni.ubc.ca  
- **Institution:** University of British Columbia
- **Here is the link to the paper:**
