# Personality Clustering, Modeling, and Analysis  
**Julian Benitez Mages & Anaelle Surprise**

---

## Project Overview

This project explores whether distinct personality groupings exist within the continuous space defined by the Big Five personality traits. Using over 1 million survey responses from the International Personality Item Pool (IPIP), we apply clustering and classification techniques to uncover latent personality structures and enable real-time predictions via a web interface.

We implemented K-Means, Gaussian Mixture Models (GMM), and DBSCAN with full grid search, integrated Factor Analysis for dimensionality reduction, and validated clusters using supervised models including logistic regression, SVMs, neural networks, and random forests.

---

## Quickstart: Run the Pipeline

To run the full demo pipeline locally:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project in demo mode (downloads data + runs best configs):
   ```bash
   python main.py
   ```

This will:
- Download the IPIP dataset automatically
- Run a shortened version of the pipeline using pre-selected model configurations
- Output clustering results, evaluation metrics, and cluster predictions

---

## Key Features

- Clustering Personality Data: Identify latent groupings using KMeans, GMM, and DBSCAN
- Dimensionality Reduction: Use Factor Analysis to enhance interpretability
- Model Evaluation: Compare clusters with Silhouette, Davies-Bouldin, and Calinski-Harabasz scores
- Cluster Prediction: Train supervised models to classify new users into clusters
- Web Deployment: Serve real-time predictions via a survey interface and cluster visualization dashboard

---

## Data Overview

- **Source:** Publicly available IPIP-50 dataset (2016–2018) via Kaggle  
- **Size:** 1,015,342 survey responses  
- **Includes:** Big Five trait responses (Likert scale), timestamp data, device info, and user metadata

---

## Web Interfaces

- **Survey Page:** Built using HTML, CSS, and D3.js — lets users take the IPIP-50 test and receive a predicted cluster in real time via a Hugging Face model API  
- **Cluster Dashboard:** Visualizes trait distributions and cluster structure using D3.js + Flask, hosted on GitHub Pages

---

## Research Background

- Grounded in the Five Factor Model (Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability)  
- Inspired by:  
  - **Gerlach et al. (2018):** Used GMM, Factor Analysis, and kernel density tests to validate clusters  
  - **Chi (2023):** Applied KMeans and supervised methods (Neural Nets, Discriminant Analysis) for predictive validation

---

## Future Work

- Apply kernel density estimation to test statistical significance of clusters  
- Compare discovered clusters to known predictors like income, marital status, or team performance  
- Explore self-perception biases in personality reporting  
- Migrate data pipeline to a scalable cloud backend (e.g., S3, GCP)  
- Expand LLM integration for natural language cluster summaries

---

## References

- Gerlach, M. et al. (2018). *Nature Human Behaviour*  
- Chi, D. (2023). *International Journal of Data Science*

---

## Conclusion

This project demonstrates how unsupervised and supervised learning can be used to explore and validate personality structures at scale. Our workflow integrates clustering, prediction, and front-end deployment to enable deeper understanding of personality data — and lay the groundwork for interactive, interpretable tools in psychology, HR, and recommendation systems.
