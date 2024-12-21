from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

SEED = 42

imputer = SimpleImputer(strategy='median')

def xgb():
    return Pipeline(steps=[
        ('imputer', imputer),
        ('regressor', XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=-1
        ))
    ])