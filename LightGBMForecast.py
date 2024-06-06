from GBDTForecast import GBDTForecast
from lightgbm import LGBMRegressor

class LightGBMForecast(GBDTForecast):

    def __init__(self, timeseries, **kwargs):
        super().__init__(timeseries, **kwargs)
        
       
    def instantiate_regressor(self, **kwargs):
        return LGBMRegressor(**self._kwargs)
