# 🔁 Adaptive Probabilistic Dialectic (APD) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A novel framework for enhancing Large Language Model reasoning through Hegelian dialectical processes combined with adaptive probabilistic methods.

## 🎯 Overview

APD implements a self-reflective dialectical process inspired by Hegelian philosophy (Thesis → Antithesis → Synthesis) with modern machine learning techniques. The framework enables LLMs to:

- **Self-criticize** and identify errors in their own reasoning
- **Generate opposing perspectives** to challenge initial solutions
- **Synthesize improved solutions** through adaptive temperature scheduling
- **Optimize computation** via Sequential Probability Ratio Test (SPRT) for early stopping

## ✨ Key Features

- **🤖 Multi-Model Support**: GPT-4o, GPT-4o-mini, GPT-4-32k with optimized configurations
- **🎛️ Adaptive Temperature Control**: Dynamic exploration-exploitation balance based on novelty metrics
- **📊 Objective Novelty Assessment**: Semantic similarity-based novelty measurement
- **⏱️ Optimal Stopping**: SPRT for efficient computation allocation
- **📈 Comprehensive Evaluation**: Quantitative and qualitative analysis tools
- **🔧 Modular Architecture**: Easily extensible components

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-username/APD.git
cd APD

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY='your-api-key-here'