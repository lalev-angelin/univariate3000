from MachineLearningForecast import MachineLearningForecast

class UnitTestMachineLearningForecast(MachineLearningForecast):
    
    def generate_forecast(self) -> None: 
        pass


forecast = UnitTestMachineLearningForecast([1,2,3,4,5,6,7,8,9,10])

inp, outp = forecast.compute_sliding_windows([1,2,3,4,5,6,7,8,9,10], input_segment_length=2, output_segment_length=2)
print(inp)
print(outp)

inp = forecast.number_input_segments(inp, starting_subperiod=2, number_of_subperiods=4)
print(inp)
