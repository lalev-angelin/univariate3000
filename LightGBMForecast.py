from GBDTForecast import GBDTForecast
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
import pandas as pd

class LightGBMForecast(GBDTForecast):

    def __init__(self, 
            timeseries: list, 
            dates: list = None, 
            forecast_horizon : int = 1, 
            lookback : int = 1,
            number_of_subperiods : int = None,
            starting_subperiod : int = 1, 
            apply_linear_detrend : bool = True,
            **kwargs):
        
        super().__init__(timeseries, 
                     dates=dates, 
                     forecast_horizon = forecast_horizon, 
                     lookback = lookback, 
                     number_of_subperiods = number_of_subperiods,
                     starting_subperiod = starting_subperiod)

        self._apply_linear_detrend = apply_linear_detrend
        self._kwargs = kwargs

        
       
    def generate_forecast(self) -> None: 
        
        if (self._apply_linear_detrend):        
            transformer = Detrender(forecaster=PolynomialTrendForecaster(degree=1))      
            timeseries = transformer.fit_transform(
                np.array(self._timeseries)).flatten().tolist()
        else: 
            timeseries = self._timeseries
        
        
        
        input_segments, output_segments = self.compute_sliding_windows(
                timeseries,
                self._lookback,
                self._forecast_horizon)
        
      
        input_segments = self.number_input_segments(input_segments)

        trainX = input_segments[:-self._forecast_horizon]
        trainY = output_segments[:-self._forecast_horizon]
         

        regressor = LGBMRegressor(
                **self._kwargs)
        
        multi_target_regressor = MultiOutputRegressor(regressor)

        multi_target_regressor.fit(trainX, trainY)

        preliminary = multi_target_regressor.predict(input_segments) 

        forecast = [np.NaN] * self._lookback
        forecast = forecast + [x[0] for x in preliminary[:-1]]
        forecast = forecast + list(preliminary[-1])
       
        assert len(forecast)==len(timeseries), "Internal error when compiling the forecast"

        if (self._apply_linear_detrend):
            self._forecast = transformer.inverse_transform(np.array(forecast)).flatten().tolist()
        else: 
            self._forecast = forecast


    def asses_forecast_mape(self) -> float:
            return self.compute_mape(self._timeseries, self._forecast, from_position = len(self._timeseries) - self._forecast_horizon)

    def asses_forecast_rmse(self) -> float:
            return self.compute_rmse(self._timeseries, self._forecast, from_position = len(self._timeseries) - self._forecast_horizon)

