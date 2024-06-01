from abc import ABC, abstractmethod
import numpy as np

####################################################################
# The abstract class representing the generic functionality of a 
# forecasting method. 

class Forecast(ABC): 
   
    __timeseries__: list = []
    __dates__: list = []
    __forecast__: list = []
    __forecast_horizon__: int = 1 

    def __init__(self, timeseries: list, dates: list = None, forecast_horizon : int = 1): 
        assert forecast_horizon > 0, "Parameter forecast_horizon must be 1 or greater."
        assert len(__timeseries__) > forecast_horizon, "The timeseries to be forecasted must contain more elements than specified in forecast_horizon"

        self.__timeseries__ = timeseries
        self.__dates__ = dates
        self.__forecast_horizon__ = forecast_horizon

        generateForecast(forecast_horizon)

    @abstractmethod
    def generateForecast() -> None:
        pass         

    def asList() -> list:
        return __forecast__

    def asNpArray() -> np.ndarray:
        return np.array(__forecast__)

