import sys
from Forecast import Forecast 

class UnitTestForecast(Forecast): 

    def generate_forecast(self): 
       self._forecast = [x+1 for x in self._timeseries]


################################################################
# START
#

forecast = UnitTestForecast([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], forecast_horizon=1)

assert forecast._forecast_horizon==1, "Error with the initialization"
assert forecast._timeseries!=None, "Error with the initialization"


assert forecast.compute_mape([1,2,3], [2,3,3], from_position=0, to_position=2)==0.75, "Error with computing MAPE 1"
assert forecast.compute_mape([1,2,3], [2,3,3])==0.5, "Error with computing MAPE 2"
assert forecast.compute_mape([1,2,3], [2,3,3], from_position=0, to_position=3)==0.5, "Error with computing MAPE 3"

assert forecast.compute_rmse([1,2,3], [2,3,3])==0.816496580927726, "Error with computing RMSE 1"
assert forecast.compute_rmse([1,2,3], [2,3,3], from_position=0, to_position=2), "Error with computing RMSE 2"

assert forecast.as_list()==[2,3,4,5,6,7,8,9,10,11]
