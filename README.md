\# Heart Disease Prediction – MLOps Assignment

## Dataset

This project uses the **UCI Heart Disease Dataset**.

### Source
UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

### How to Obtain the Dataset

1. Download the dataset from the UCI repository.
2. Place the raw file in:notebooks/data/raw/heart_disease.csv
3. Run the preprocessing script:
   bash
   python src/models/preprocess_data.py
4.This will generate the cleaned dataset at:
   notebooks/data/processed/heart_disease_cleaned.csv
5.This processed dataset is used for model training, evaluation, and API inference.








End-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset.



