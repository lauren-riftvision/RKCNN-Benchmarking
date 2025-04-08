# Random K Conditional Nearest Neighbor (RKCNN) Implementation and Benchmarking

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) 

## Overview

This repository contains the implementation of the Random K Conditional Nearest Neighbor (RKCNN) classification algorithm, a novel variant of KNN proposed by Jiaxuan Lu and Hyukjun Gweon (refer to the [Random Conditional K-Nearest Neighbors (RKCNN) Paper](https://peerj.com/articles/cs-2497/)).

In this project, RKCNN is implemented from the mathematical description and pseudocode provided in the paper (note: this implementation may differ slightly from the researchers' provided Python code). The algorithm is benchmarked against traditional KNN, KCNN, XGBoost, and BERT on a high-dimensional, noisy dataset (the 20 Newsgroups dataset with Sentence-BERT embeddings).

## Key Features

* Implementation of the RKCNN algorithm.
* Implementation of the KCNN algorithm.
* Benchmarking against several established classification models.
* Evaluation of model performance using standard classification metrics.
* Per-class evaluation of the RKCNN model compare to BERT.

## Repository Contents

* `RKCNN_Code.py`: The main Python script containing the RKCNN and KCNN implementations, data loading, benchmarking, and evaluation code (used within the Jupyter Notebook).
* `requirements.txt`: A list of Python dependencies required to run the code and the Jupyter Notebook.
* `RKCNN_Benchmarking.ipynb`: The Jupyter Notebook containing the full analysis, including benchmarking and per-class evaluation.
* `README.md`: This file, providing an overview of the project.

## Getting Started

### Prerequisites

* Python 3.9 (or a compatible version).
* Jupyter Notebook (`pip install notebook`).
* The Python dependencies listed in `requirements.txt`. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```
* **Dataset:** The `RKCNN_20NewsProcessed.csv` dataset is not included directly in this repository due to its size. You will need to download it from the link provided in the "Running the Code" section.

### Running the Code

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd [repository name]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the dataset:**
    The `RKCNN_20NewsProcessed.csv` dataset is available for download from [[Google Drive](https://drive.google.com/file/d/1HiJTX-LuHD5e_qmgaeehxfBH5XjYaX_n/view?usp=sharing)]. Please download this file and place it in the same directory as the Jupyter Notebook (`RKCNN_Benchmarking.ipynb`).

4.  **Open and run the Jupyter Notebook:**
    Navigate to the repository directory in your terminal and run:
    ```bash
    jupyter notebook
    ```
    This will open the Jupyter Notebook interface in your web browser. Open the `RKCNN_Benchmarking.ipynb` file and run the cells to execute the code and see the results. The notebook is configured to load the `RKCNN_20NewsProcessed.csv` file from the local directory.

## Results

The benchmarking results presented in the Jupyter Notebook indicate that RKCNN achieves competitive performance, standing strong not only against traditional KNN but also exhibiting a competitive edge against powerful transformer models like BERT. Its innovative approach, leveraging separation scores and a weighted aggregation of predictions from random feature subsets, appears to be an effective strategy for mitigating bias, variance, and overfitting, particularly in high-dimensional datasets. Refer to the notebook for detailed results and visualizations.

### Metrics for RKCNN (20News Dataset):

Optimal Parameters:
k=2, h=100, r=50, m=0.3

| Metric    | Value    |
|-----------|----------|
| Accuracy  | 0.747215 |
| Precision | 0.758503 |
| Recall    | 0.747215 |
| F-1 Score | 0.746383 |

RKCNN outperforms BERT's reported accuracy of approximately 74.67% on the same dataset.

Refer to `RKCNN_Benchmarking.ipynb` for detailed per-class performance metrics and visualizations.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). 


## Acknowledgements

* The researchers Jiaxuan Lu and Hyukjun Gweon for their work on the RKCNN algorithm. [Original Research Paper Code](https://dfzljdn9uc3pi.cloudfront.net/2025/cs-2497/1/RandomkCNN.py)
* The creators and maintainers of the 20 Newsgroups dataset and the Sentence-BERT embeddings.
* The developers of the Python libraries used in this project (scikit-learn, pandas, etc.).
