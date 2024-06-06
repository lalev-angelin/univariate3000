from GBDTForecast import GBDTForecast
from catboost import CatBoostRegressor

class CatBoostForecast(GBDTForecast):

    def __init__(self, timeseries, **kwargs):
        super().__init__(timeseries, **kwargs)
        
       
    def instantiate_regressor(self, **kwargs):
        return CatBoostRegressor(**self._kwargs)
