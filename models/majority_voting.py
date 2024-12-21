from sklearn.ensemble import VotingRegressor
from models.xgb import pipeline_xgb
from models.random_forest import pipeline_rf
from models.lgbm import pipeline_lgbm

def majority_voting():
    return VotingRegressor(estimators=[
        ('lgbm', pipeline_lgbm),
        ('xgb', pipeline_xgb),
        ('rf', pipeline_rf)
    ])


