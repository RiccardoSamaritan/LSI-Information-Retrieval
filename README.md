# SVD and Latent Semantic Indexing

A study and implementation project of Singular Value Decomposition (SVD) and Latent Semantic Indexing (LSI) techniques for improving information retrieval systems.
This project was realized by:
  - [Giovanni Oro](https://github.com/GiovanniOro666)
  - [Riccardo Samaritan](https://github.com/RiccardoSamaritan)
for the Information Retrieval exam at University Of Trieste

## Project Overview

This project explores the application of Singular Value Decomposition (SVD) in the context of latent semantic indexing to solve fundamental problems of traditional vector space models in information retrieval.

### Problems Addressed

- **Synonymy**: Different words with the same meaning are mapped to different points in the vector space
- **Polysemy**: The same word can have different meanings, but the vector space model considers them all equal

### Proposed Solution

**Latent Semantic Indexing (LSI)** uses SVD to:
- Map terms to abstract concepts
- Represent documents as combinations of these concepts
- Reduce dimensionality while preserving the most important semantic information

## Implemented Theoretical Concepts

### 1. Singular Value Decomposition (SVD)

For a term-document matrix **C** of dimension M × N:

```
C = UΣV^T
```

Where:
- **U**: M × r matrix with orthonormal eigenvectors of CC^T
- **Σ**: r × r diagonal matrix with singular values
- **V**: N × r matrix with orthonormal eigenvectors of C^TC

### 2. Low Rank Approximation

**Objective**: Reduce the space occupied by a matrix by reducing its rank, minimizing error.

**Procedure**:
1. Calculate SVD of matrix C
2. Set k < r as desired rank
3. Construct Σₖ by zeroing the smallest singular values
4. Obtain: Cₖ = UₖΣₖVₖ^T

### 3. Latent Semantic Indexing

**Advantages**:
- Solves Synonymy and Polysemy problems
- Improves retrieval quality
- Can be used for clustering, synonym search, query expansion

**Disadvantages**:
- SVD computation is computationally expensive
- Adding new documents requires complete SVD recalculation

## Project Structure

```
├── README.md
├── src/
│   ├── data_processor.py       # To process data needed for the LSI system
│   ├── lsi.py                  # Puts together all the functionalites of the system
│   ├── lsi_core.py             # Core LSI operations including SVD decomposition and concept analysis
│   └── query_processor.py      # Handles query preprocessing and vector creation
├── data/           # This folder includes the data to test the system
└── main.ipynb      # Jupyter Notebook with execution and results for the LSI system
└── time_dataset.ipynb      # LSI applied to TIME dataset
```

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd svd-latent-semantic-indexing

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
```

## Mathematical Foundation

### Term-Concept Matrix (U)
- Entry (i,j) represents how much term i is represented by concept j
- Concepts are linear combinations of terms

### Concept Weights (Σ)
- Diagonal values represent concept importance
- Higher values indicate greater concept significance

### Document-Concept Matrix (V^T)
- Entry (i,j) represents how much document i contains concept j
- Documents are mapped to latent space through this matrix

### Query Processing
For a query vector q, the mapping to latent space is:
```
q̂ = Σ⁻¹U^T q
```

## Features

- **Dimensionality Reduction**: Reduces computational complexity while preserving semantic relationships
- **Semantic Similarity**: Finds documents with similar meanings even with different vocabularies
- **Noise Reduction**: Filters out less important terms and concepts
- **Cosine Similarity**: Uses cosine similarity for document ranking
