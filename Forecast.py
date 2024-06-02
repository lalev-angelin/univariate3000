from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


####################################################################
# The abstract class representing the generic functionality of a 
# forecasting method. 

class Forecast(ABC): 
   
    __timeseries__: list = []
    __dates__: list = []
    __forecast__: list = []
    __forecast_horizon__: int = 1 
    __number_of_subperiods__: int = None
    __starting_subperiod__: int = 0 

    def __init__(self, timeseries: list, dates: list = None, forecast_horizon : int = 1, 
            number_of_subperiods = None, starting_subperiod = 0): 
        assert forecast_horizon > 0, "Parameter forecast_horizon must be 1 or greater."
        assert len(__timeseries__) > forecast_horizon, "The timeseries to be forecasted must contain more elements than specified in forecast_horizon"

        self.__timeseries__ = timeseries
        self.__dates__ = dates
        self.__forecast_horizon__ = forecast_horizon
        self.__number_of_subperiods__ = number_of_subperiods
        self.__starting_subperiod__ = starting_subperiod

        generateForecast(forecast_horizon)

    @abstractmethod
    def generateForecast() -> None:
        pass         

    def asList() -> list:
        return __forecast__

    def asNpArray() -> np.ndarray:
        return np.array(__forecast__)

    def computeMAPE(from_position : int) -> float: 
        if from_position is None:
            return mean_absolute_percentage_error(__timeseries__, __forecast__)
        else: 
            return mean_absolute_percentage_error(__timeseries__[-position:], __forecast__[-position:])
         
    def computeRMSE(from_position : int) -> float:
        if from_position is None: 
            return math.sqrt(mean_squared_error(__timeseries__, __forecast__))
        else: 
            return math.sqrt(mean_squared_error(__timeseries__[-position:], __forecast__[-position:]))
