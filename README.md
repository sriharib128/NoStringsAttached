# Knowledge Graph Debiasing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.0+-blueviolet.svg)](https://pytorch-geometric.readthedocs.io/)

## Overview

This project implements a framework for detecting and mitigating gender bias in knowledge graph embeddings. Using the FB15k-237 dataset (a subset of Freebase), we analyze how gender associations can influence the way professions and other entities are represented in embedding space. The framework includes:

1. A custom TransE implementation with debiasing mechanisms
2. Bias measurement methodology
3. Visualization tools for bias analysis
4. Comparative evaluation of standard vs. debiased embeddings

![Bias Reduction](RSAI.png)

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Debiasing Techniques](#debiasing-techniques)
- [Evaluation](#evaluation)
- [License](#license)

## Installation

### Requirements

Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Required packages are listed in `requirements.txt`:

```
torch>=1.8.0
torch-geometric>=2.0.0
numpy>=1.19.5
pandas>=1.2.4
matplotlib>=3.4.2
scikit-learn>=0.24.2
tqdm>=4.61.0
```

## Dataset

The project uses the FB15k-237 dataset, a subset of Freebase commonly used for knowledge graph completion tasks. Our analysis focuses on gender-profession relationships.

Dataset files:
- `FB15k/raw/entities.dict`: Entity ID to index mapping
- `FB15k/raw/relations.dict`: Relation ID to index mapping
- `FB15k_mid2name.txt`: Maps Freebase MIDs to human-readable names
- `gen2prof_fair_all.txt`: Gender-profession relationship data

## Usage

### Data Preparation

```bash
# Download and prepare the FB15k-237 dataset
python prepare_data.py
```

### Training the TransE Model

#### Standard TransE (with no debiasing)

```bash
python train.py --model standard
```

#### Debiased TransE

```bash
python train.py --model debiased --lambda_eq 0.1 --lambda_ortho 0.1 --lambda_adv 0.1
```

### Bias Analysis

```bash
# Generate bias score analysis
python analyze_bias.py --input_model model_weights.pt --output_file bias_scores.json

# Visualize bias scores
python visualize_bias.py --input_file bias_scores.json
```

## Model Architecture

The project implements the TransE knowledge graph embedding model with additional debiasing components:

1. **Base TransE**: Represents entities and relations in the same vector space, where (head + relation ≈ tail)
2. **Gradient Reversal Layer**: Used for adversarial debiasing
3. **Orthogonality Constraints**: Enforces orthogonality between sensitive attributes and target attributes
4. **Equalized Odds Module**: Ensures similar treatment across sensitive attribute groups

## Debiasing Techniques

This project implements several debiasing techniques:

1. **Adversarial Debiasing**: Using gradient reversal to prevent the model from learning gender information
2. **Orthogonality Constraints**: Making profession embeddings orthogonal to gender embeddings
3. **Equalized Odds**: Ensuring similar distances between entities across gender groups

Hyperparameters control the strength of each debiasing component:
- `lambda_eq`: Weight for equalized odds loss
- `lambda_ortho`: Weight for orthogonality constraint
- `lambda_adv`: Weight for adversarial component

## Evaluation

We evaluate bias in knowledge graph embeddings using several metrics:

1. **Bias Score**: Measures how much gender information affects profession embeddings
2. **Link Prediction Performance**: Ensures debiasing doesn't significantly harm task performance
3. **Gender Profession Gap**: Analyzes differences in prediction scores for male vs. female entities

## License

This project is licensed under the MIT License - see the LICENSE file for details.
