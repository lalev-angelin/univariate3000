from GBDTForecast import GBDTForecast
from xgboost import XGBRegressor

class XGBoostForecast(GBDTForecast):

    def __init__(self, timeseries, **kwargs):
        super().__init__(timeseries, **kwargs)
        
       
    def instantiate_regressor(self, **kwargs):
        return XGBRegressor(**self._kwargs)
