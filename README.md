# Child Mind Institute Solution INT3405E

## Introduction
This project aims to address the **Problematic Internet Use** challenge provided by the Child Mind Institute on Kaggle. The goal is to build robust machine learning models to analyze and predict internet usage behavior based on structured data.

---

## Project Structure

Child Mind Institute/

├─ configs/  
│  ├─ config.py  
├─ models/  
│  ├─ random_forest.py  
│  ├─ xgb.py  
│  ├─ kappa.py  
│  ├─ lgbm.py  
│  ├─ majority_voting.py  
├─ notebooks/  
├─ networks/  
│  ├─ auto_encoder.py  
├─ scripts/  
│  ├─ scripts.py  
│  ├─ train.py  
├─ README.md  

---

### Folder Descriptions
- **configs/**: Contains configuration files (e.g., model parameters, environment setup).
- **models/**: Scripts for machine learning models, including Random Forest, XGBoost, LightGBM, and Majority Voting.
- **notebooks/**: Jupyter notebooks for data exploration, analysis, and preparing final submissions.
- **networks/**: Contains scripts for neural network models like Autoencoder.
- **scripts/**: Includes utility scripts like `train.py` for model training and `scripts.py` for other helper functions.

---

## How to Implement

### Step 1: Download the Notebook
Download the `notebooks/final_submission.ipynb` file, which serves as the primary notebook for running and preparing submissions.

### Step 2: Join the Kaggle Competition
Visit the [Child Mind Institute Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) competition page. Join the competition and create a team to participate.

### Step 3: Run the Code
1. Open the `final_submission.ipynb` notebook.
2. Run all code cells to:
   - Load and preprocess data.
   - Train the machine learning models.
   - Evaluate model performance.
   - Generate the final `submission.csv`.

### Step 4: Generate the Submission
After executing all notebook cells, the `submission.csv` file will be generated. Use this file for submission on Kaggle.

### Step 5: Submit the Submission
Log in to Kaggle and upload your `submission.csv` file using the **Submit** button.

---

## Models Used
The following models are implemented in this project:

1. **Random Forest with GridSearchCV and Pipeline**:
   - A Random Forest model with hyperparameter tuning via GridSearchCV, integrated into a Pipeline for streamlined training and evaluation.

2. **LightGBM**:
   - Light Gradient Boosting Machine, known for its speed and efficiency in gradient boosting.

3. **XGBoost**:
   - Extreme Gradient Boosting, widely used for structured data problems due to its high performance.

4. **Majority Voting**:
   - An ensemble method combining predictions from multiple models to enhance robustness and accuracy.

---

## Contributors
**Team Members**:
- **K-ICM**
  - Do Dinh Dung - 22028169
  - Vu Viet Hung - 22028124
  - Le Van Luong - 22028040

---

## Installation

Install the required packages using the command below:
```sh
pip install -r requirements.txt
