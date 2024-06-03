from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math

####################################################################
# The abstract class representing the generic functionality of a 
# forecasting method. 

class Forecast(ABC): 
   
    _timeseries : list = []
    _dates : list = []
    _forecast : list = []
    _forecast_horizon : int = 1 
    _number_of_subperiods : int = None
    _starting_subperiod : int = 0 

    def __init__(self, timeseries: list, dates: list = None, forecast_horizon : int = 1, 
            number_of_subperiods : int = None, starting_subperiod : int = 0): 

        assert forecast_horizon > 0, "Parameter forecast_horizon must be 1 or greater."
        assert len(timeseries) > forecast_horizon, "The timeseries to be forecasted must contain more elements than specified in forecast_horizon"

        self._timeseries = timeseries
        self._dates = dates
        self._forecast_horizon = forecast_horizon
        self._number_of_subperiods = number_of_subperiods
        self._starting_subperiod = starting_subperiod

        self.generate_forecast()
    
    @abstractmethod
    def generate_forecast(self) -> None:
        pass         

    def as_list(self) -> list:
        return self._forecast

    def as_nparray(self) -> np.ndarray:
        return np.array(self._forecast)


    def compute_mape(self, actual: list, forecast: list, from_position : int = None, to_position: int = None) -> float: 

        assert len(actual) == len(forecast), "Length of the actual and forecast lists must be equal"

        if from_position is None:
            from_position = 0

        if to_position is None: 
            to_position = len(actual)

        return mean_absolute_percentage_error(actual[from_position:to_position], forecast[from_position:to_position])
 

    def compute_rmse(self, actual: list, forecast: list, from_position : int = None, to_position: int = None) -> float: 

        assert len(actual) == len(forecast), "Length of the actual and forecast lists must be equal"

        if from_position is None:
            from_position = 0

        if to_position is None: 
            to_position = len(actual)

        return math.sqrt(mean_squared_error(actual[from_position:to_position], forecast[from_position:to_position]))

         
