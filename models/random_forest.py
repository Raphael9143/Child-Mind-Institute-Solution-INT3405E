from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import config

imputer = SimpleImputer(strategy='median')

def random_forest():
    return Pipeline(steps=[
        ('imputer', imputer),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=config.SEED,
            n_jobs=-1
        ))
    ])