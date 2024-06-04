from GBDTForecast import GBDTForecast
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

class LightGBMForecast(GBDTForecast):

    def __init__(self, 
            timeseries: list, 
            dates: list = None, 
            forecast_horizon : int = 1, 
            lookback : int = 1,
            number_of_subperiods : int = None,
            starting_subperiod : int = 1, 
            **kwargs):
        
        super().__init__(timeseries, 
                     dates=dates, 
                     forecast_horizon = forecast_horizon, 
                     lookback = lookback, 
                     number_of_subperiods = number_of_subperiods,
                     starting_subperiod = starting_subperiod)

        self._kwargs = kwargs

        
       
    def generate_forecast(self) -> None: 
        
        input_segments, output_segments = self.compute_sliding_windows(
                self._timeseries,
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

        self._forecast = [np.NaN] * self._lookback
        self._forecast = self._forecast + [x[0] for x in preliminary[:-1]]
        self._forecast = self._forecast + list(preliminary[-1])
       
        assert len(self._forecast)==len(self._timeseries), "Internal error when compiling the forecast"

    def asses_forecast_mape(self) -> float:
            self.compute_mape(self._timeseries, self._forecast, from_position = len(self._timeseries) - self._forecast_horizon)

    def asses_forecast_rmse(self) -> float:
            self.compute_rmse(self._timeseries, self._forecast, from_position = len(self._timeseries) - self._forecast_horizon)

