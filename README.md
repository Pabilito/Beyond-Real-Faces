# Replication Package: Beyond Real Faces

[![arXiv](https://img.shields.io/badge/arXiv-2510.17372-b31b1b.svg)](https://arxiv.org/abs/2510.17372)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the replication package for **"Beyond Real Faces: Synthetic Datasets Can Achieve Reliable Recognition Performance without Privacy Compromise"**. This package enables researchers to evaluate synthetic facial recognition datasets by computing embeddings, analyzing distribution characteristics, and comparing against the CASIA-WebFace benchmark.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Computing Embeddings](#1-computing-embeddings)
  - [2. Mated vs Non-Mated Analysis](#2-mated-vs-non-mated-analysis)
  - [3. CASIA Comparison](#3-casia-comparison)
- [Data](#data)
- [Citation](#citation)

## Overview

This package provides tools to:
- Compute facial embeddings using pretrained ArcFace models
- Analyze mated (same identity) vs non-mated (different identity) similarity distributions
- Compare synthetic datasets against CASIA-WebFace to detect potential identity leakage
- Visualize similarity distributions and closest matching samples

## Installation

### 1. Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate [ENV_NAME]
```

### 2. Download Pretrained Model

Download the ArcFace R100 model trained on MS1MV3:
- Source: [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- Save to: `./Models/ArcFace_R100_MS1MV3.pth`

**Note:** If using a different model, modify `Embedding_Computation/analyze_embeddings.py` accordingly.

### 3. Dataset Organization

Place your datasets in the `./Dataset/` directory. Due to size constraints, datasets and pretrained weights are not included in this repository.

## Usage

> **Note:** Replace placeholder parameter values (shown in UPPERCASE) with actual values from your setup.

### 1. Computing Embeddings

Generate embedding vectors for your facial recognition dataset:

```bash
cd Embedding_Computation
python compute_embeddings_to_json.py
```

**Output:** JSON file containing facial embeddings for each image in the dataset.

---

### 2. Mated vs Non-Mated Analysis

Analyze the distribution of similarity scores between images of the same identity (mated) versus different identities (non-mated).

#### Basic Distribution Plot

```bash
cd Mated_vs_non-mated
python analyze_embeddings.py INPUT.json --n_comparisons 10000
```

**Parameters:**
- `INPUT.json`: Embeddings file from step 1
- `--n_comparisons`: Number of comparison pairs to sample

**Output:** Histogram showing mated and non-mated similarity distributions

![Vec2Face_MatedVsNonmated_1000000](https://github.com/user-attachments/assets/9eb631f9-2528-4a47-b8e8-a7a636c5041d)

#### Detailed Metrics

For additional metrics including EER, AUC, and statistical measures:

```bash
python compute_similarities.py INPUT.json --n_comparisons 10000
python analyze_embeddings_with_metrics.py
```

**Output:** Enhanced distribution plot with performance metrics

![CemiFace_MatedVsNonmated_2000000](https://github.com/user-attachments/assets/67b19f7d-db34-4458-a9ca-e0a8b8e1850f)

---

### 3. CASIA Comparison

Evaluate potential identity leakage by comparing your synthetic dataset against the real-world CASIA-WebFace dataset.

#### Option A: Distribution Plot

Find best matches in CASIA and visualize the similarity distribution:

```bash
cd CASIA_compare
python compare_embeddings.py INPUT.json --n_comparisons 10000 --casia_file CASIA.json
```

**Parameters:**
- `INPUT.json`: Your dataset embeddings
- `CASIA.json`: CASIA-WebFace embeddings
- `--n_comparisons`: Number of samples to compare

**Output:** Distribution of best-match similarity scores

![Vec2Face_BestMatches_10000](https://github.com/user-attachments/assets/4f6ca291-473e-4e13-8421-c81a3fce2941)

#### Option B: Save Similarity Scores

Export similarity values without plotting:

```bash
python compute_similarities.py INPUT.json --n_comparisons 10000 --casia_file CASIA.json
```

**Output:** `SimilarityScores/INPUT_BestMatches_10000.txt`

#### Option C: Visual Comparison of Closest Samples

Generate side-by-side visualizations of the most similar image pairs:

1. Find closest matches:
```bash
./find_closest.sh
```
**Output:** JSON file listing closest sample pairs

2. Extract image paths:
```bash
./get_paths.sh
```
**Output:** `SIMILAR_IMAGES.csv`

3. Plot comparisons (supports `.zip` and `.tar.gz` archives):
```bash
python plot_similar.py SIMILAR_IMAGES.csv --casia-file CASIA.zip --other-file OTHER.zip --samples 5
```

**Parameters:**
- `--samples`: Number of closest pairs to visualize

**Output:** Grid visualization comparing synthetic faces with their closest CASIA matches

![comparison_IDiffFace_5_samples](https://github.com/user-attachments/assets/24315218-816c-4c1c-a506-804a9775152f)

## Data

Pre-computed similarity metrics and closest sample analyses are available in the `./Metrics/` folder.

## Citation

If you use this replication package, please cite:

```bibtex
@article{borsukiewicz2025beyond,
  title={Beyond Real Faces: Synthetic Datasets Can Achieve Reliable Recognition Performance without Privacy Compromise},
  author={Borsukiewicz, Pawe{\l} and Boutros, Fadi and Olatunji, Iyiola E and Beumier, Charles and Ouedraogo, Wendk{\^u}uni C and Klein, Jacques and Bissyand{\'e}, Tegawend{\'e} F},
  journal={arXiv preprint arXiv:2510.17372},
  year={2025}
}
```

**Paper:** [https://arxiv.org/pdf/2510.17372](https://arxiv.org/pdf/2510.17372)


## Contact

For questions or issues, please contact the authors using emails provided in the paper.
