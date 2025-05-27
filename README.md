# Multimodal Network and Textual Analysis of Reddit Engagement During Exam Seasons

_An investigation of factors driving student engagement on r/mcgill and r/concordia Reddit communities during exam periods using social network analysis, NLP, and graph neural networks._

## Table of Contents

1. [Description](#description)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Installation](#installation)  
5. [Data Sources](#data-sources)  
6. [Usage](#usage)  
7. [Project Structure](#project-structure)  
8. [Configuration](#configuration)

---

## Description

This project analyzes Reddit activity from r/mcgill and r/concordia during final exam months (April & December 2021–2022) to identify drivers of high engagement (post scores and replies). We combine:

- **Network Analysis**: Build reply graphs to compute centrality and detect communities.  
- **Textual Analysis**: Apply topic modeling (LDA) and sentiment analysis (VADER) to uncover discussion themes and tone.  
- **Predictive Modeling**: Use Graph Neural Networks (GCN, GAT, GraphSAGE) with structural and textual features to predict high-impact posts and users.

## Key Features

- **Data Integration**: Merge comments and submissions into directed user reply networks.  
- **Centrality Metrics**: Degree, closeness, betweenness, eigenvector, and Katz centrality for influence analysis.  
- **Community Detection**: Louvain algorithm to uncover thematic subgroups.  
- **NLP Processing**: Tokenization, lemmatization, stop-word removal, sentiment scoring, and LDA topic extraction.  
- **GNN Models**: Train and evaluate GCN, GAT, and GraphSAGE to classify above-median engagement posts.  
- **Visualization**: Network plots, choropleths of centrality, topic word clouds, and performance curves.

## Tech Stack

- **Language:** Python 3.8+  
- **Notebooks:** Jupyter Notebook (`INSY670_Group3_Code_Concordia.ipynb`, `INSY670_Group3_Code_McGill.ipynb`)  
- **Libraries:** pandas, networkx, numpy, scikit-learn, matplotlib, seaborn, gensim, vaderSentiment, torch_geometric, folium  

## Installation

```bash
git clone https://github.com/your-username/insy670-reddit-engagement.git
cd insy670-reddit-engagement
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Data Sources

- **Reddit Data**: CSVs of comments and submissions for r/mcgill and r/concordia (April & December 2021–2022).  
- **Mappings**: `precinct-puma.csv`-style mapping if needed for geospatial context (optional).  
- **External Metrics**: Pre-computed sentiment and topic distributions saved as CSV for fast loading.

> Place all data files in the `data/` directory before running analysis.

## Usage

1. **Preprocess data and build networks**:  
   ```bash
   jupyter notebook INSY670_Group3_Code_Concordia.ipynb
   ```  
2. **Run centrality & community detection** in respective notebooks.  
3. **Perform NLP analysis**: topic modeling and sentiment scoring.  
4. **Train GNN models** for engagement prediction and evaluate accuracy.  
5. **Review visualizations** in `figures/`: centrality distributions, topic clouds, loss curves, and accuracy plots.

## Project Structure

```
insy670-reddit-engagement/
├── README_INSY670.md                   # This file
├── requirements.txt                    # Project dependencies
├── data/                               # Raw and processed datasets
│   ├── mcgill_comments.csv
│   ├── mcgill_submissions.csv
│   ├── concordia_comments.csv
│   └── concordia_submissions.csv
├── INSY670_Group3_Code_Concordia.ipynb # Analysis notebook for Concordia data
├── INSY670_Group3_Code_McGill.ipynb    # Analysis notebook for McGill data
├── figures/                            # Generated plots and maps
│   ├── centrality_plots.png
│   ├── topic_wordclouds.png
│   └── gnn_performance.png
└── scripts/                            # Utility scripts
    └── preprocess.py
```

## Configuration

- **Exam Months**: Default months set to April and December; modify in `preprocess.py`.  
- **GNN Hyperparameters**: Learning rate, epochs, and architecture choice can be adjusted in the model training cells.  
- **Topic Model Parameters**: Number of topics and passes configurable in the LDA notebook.  
