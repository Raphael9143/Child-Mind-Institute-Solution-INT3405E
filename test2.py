import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

try:
    train_ds = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv', index_col='id')
    test_ds = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv', index_col='id')
    sample_submission = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')
except FileNotFoundError as e:
    print("File not found. Please check the file path.")
    raise e

if train_ds['sii'].isnull().any():
    print("Missing values detected in 'sii'.")
    train_ds = train_ds.dropna(subset=['sii'])

categorical_features = ['PCIAT-Season']
missing_columns = [col for col in categorical_features if col not in train_ds.columns or col not in test_ds.columns]

if missing_columns:
    print(f"The following columns are missing: {missing_columns}")
    for col in missing_columns:
        if col not in train_ds.columns:
            train_ds[col] = np.nan
        if col not in test_ds.columns:
            test_ds[col] = np.nan

season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}

for season in categorical_features:
    if season in train_ds.columns:
        train_ds[season] = train_ds[season].replace(season_mapping)
    if season in test_ds.columns:
        test_ds[season] = test_ds[season].replace(season_mapping)

X = train_ds.drop(columns=['sii'])
y = train_ds['sii']

numeric_cols = X.select_dtypes(include=['number']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

non_numeric_cols = X.select_dtypes(exclude=['number']).columns
X[non_numeric_cols] = X[non_numeric_cols].fillna(X[non_numeric_cols].mode().iloc[0])

if X.isnull().any().any():
    print("Missing values detected in features.")
    X = X.fillna(X.mean())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

test_ds = test_ds.fillna(test_ds.mean())  
test_predictions = model.predict(test_ds)

sample_submission['sii'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
