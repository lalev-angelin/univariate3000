from MachineLearningForecast import MachineLearningForecast
from abc import abstractmethod
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.detrend import ConditionalDeseasonalizer
from sktime.forecasting.trend import PolynomialTrendForecaster
import pandas as pd


class GBDTForecast(MachineLearningForecast):

    def __init__(self,
            timeseries: list,
            dates: list = None,
            forecast_horizon : int = 1,
            lookback : int = 1,
            number_of_subperiods : int = None,
            starting_subperiod : int = 1,
            apply_linear_detrend : bool = True,
            apply_deseasonalizer : bool = False,
            **kwargs):

        super().__init__(timeseries,
                     dates=dates,
                     forecast_horizon = forecast_horizon,
                     lookback = lookback,
                     number_of_subperiods = number_of_subperiods,
                     starting_subperiod = starting_subperiod)

        self._apply_linear_detrend = apply_linear_detrend
        self._apply_deseasonalizer = apply_deseasonalizer
        self._kwargs = kwargs

    @abstractmethod
    def instantiate_regressor(self, **kwargs): 
        pass


    def generate_forecast(self) -> None:

        timeseries = self._timeseries

        if self._apply_deseasonalizer and self._number_of_subperiods is not None and self._number_of_subperiods>0:
            deformer = ConditionalDeseasonalizer(sp=self._number_of_subperiods)
            timeseries = deformer.fit_transform(np.array(timeseries)).flatten().tolist()


        if self._apply_linear_detrend:
            transformer = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
            timeseries = transformer.fit_transform(
                np.array(timeseries)).flatten().tolist()


        input_segments, output_segments = self.compute_sliding_windows(
                timeseries,
                self._lookback,
                self._forecast_horizon)


        input_segments = self.number_input_segments(input_segments)

        trainX = input_segments[:-self._forecast_horizon]
        trainY = output_segments[:-self._forecast_horizon]


        regressor = self.instantiate_regressor(**self._kwargs)

        multi_target_regressor = MultiOutputRegressor(regressor)

        #print(self._timeseries, "\n")
        multi_target_regressor.fit(trainX, trainY)

        preliminary = multi_target_regressor.predict(input_segments)

        forecast = [np.NaN] * self._lookback
        forecast = forecast + [x[0] for x in preliminary[:-1]]
        forecast = forecast + list(preliminary[-1])

        assert len(forecast)==len(timeseries), "Internal error when compiling the forecast"

        if self._apply_linear_detrend:
            forecast = transformer.inverse_transform(np.array(forecast)).flatten().tolist()

        if self._apply_deseasonalizer:
            forecast = deformer.inverse_transform(np.array(forecast)).flatten().tolist()

        self._forecast = forecast
        return

