from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
import config

imputer = SimpleImputer(strategy='median')

params = {
    'objective'       : 'l2',
    'verbosity'       : -1,
    'lambda_l1'       : 0.005116829730239727,
    'lambda_l2'       : 0.0011520776712645852,
    'learning_rate'   : 0.02376367323636638,
    'max_depth'       : 5,
    'num_leaves'      : 207,
    'colsample_bytree': 0.7759862336963801,
    'colsample_bynode': 0.5110355095943208,
    'bagging_fraction': 0.5485770314992224,
    'bagging_freq'    : 7,
    'min_data_in_leaf': 78,
}

def lgbm():
    return Pipeline(steps=[
        ('imputer', imputer),
        ('regressor', LGBMRegressor(
            n_estimators=100,
            random_state=config.SEED,
            n_jobs=-1,
            **params
        ))
    ])
