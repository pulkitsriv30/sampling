# Sampling Assignment

This project demonstrates the impact of different sampling techniques on the accuracy of various machine learning models using a balanced credit card dataset.

## Project Overview

The goal of this assignment is to:
1.  **Balance the Dataset**: Convert the imbalanced `Creditcard_data.csv` into a balanced dataset using SMOTE (Synthetic Minority Over-sampling Technique).
2.  **Apply Sampling Techniques**: Create five different samples using the following techniques:
    *   Simple Random Sampling
    *   Systematic Sampling
    *   Stratified Sampling
    *   Cluster Sampling
    *   Bootstrap Sampling
3.  **Evaluate Models**: Train and test five different machine learning models on each sample:
    *   Logistic Regression
    *   Decision Tree
    *   Random Forest
    *   Support Vector Machine (SVM)
    *   K-Nearest Neighbors (KNN)
4.  **Compare Results**: Determine which sampling technique yields the highest accuracy for each model.

## Requirements

Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## files

*   `sampling102303803.py`: The main Python script that performs data processing, sampling, and model evaluation.
*   `Creditcard_data.csv`: The input dataset (must be present in the same directory).
*   `sampling_results.csv`: The output file containing the accuracy table.

## Usage

Run the script using the following command:

```bash
python sampling102303803.py
```

## Results

After execution, the script will:
*   Print the class distribution before and after balancing.
*   Display the accuracy of each model for every sampling technique.
*   Identify the best sampling technique for each model.
*   Save the detailed results to `sampling_results.csv`.

## Methodology

*   **Balancing**: `SMOTE` is used to oversample the minority class to achieve a 50-50 balance.
*   **Sample Size**: calculated using Cochran's formula with a 95% confidence level and 5% margin of error.
*   **Evaluation**: Models are trained on the generated samples and tested on a hold-out set (20% of the balanced data) to ensure fair comparison.
