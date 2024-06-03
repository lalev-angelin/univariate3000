from Forecast import Forecast
from abc import abstractmethod 
import sys

# The abstract class representing the generic functionality of a
# forecasting method.

class MachineLearningForecast(Forecast):

    __lookback__: int = 1


    #####################################################################
    # Constructor 

    def __init__(self, timeseries: list, dates: list = None, forecast_horizon : int = 1, number_of_subperiods=None,
                starting_subperiod : int = 1, lookback: int = 1):
        assert lookback>0, "Lookback must be value greater than 1"
        self.__lookback__ = lookback
         
        super().__init__(timeseries, dates=dates, forecast_horizon=forecast_horizon, 
                number_of_subperiods=number_of_subperiods, starting_subperiod=starting_subperiod)



    ######################################################################
    # Trees get vectors as input  This requires
    # input data of our timeseries to be organized in such
    # overlapping segments/vectors.

    def compute_sliding_windows(self, datalist: list, input_segment_length: int, output_segment_length: int = 1) -> list:

        # We start by spliting the whole row into "training" windows of
        # input_segment_length. Keep in mind that if you pick  list of say 7
        # elements  and 3 input, you will have 5 such segments.
        # [1 2 3 4 5 6 7] -> [1 2 3], [2 3 4], [3 4 5], [4 5 6], [5 6 7]
        # This is because you can get as high as 5 and have the two last
        # numbers to compose your last segment.
        # So the higest position from which segment may be made is
        # the length of the row minus the input_length, decreased by one.
        input_segments = []

        for start in range(0, len(datalist)-(input_segment_length-1)):
            input_segments.append(datalist[start:start+input_segment_length])

        # Exactly the same procedure for the output segments

        output_segments = []

        for start in range(0, len(datalist)-(output_segment_length-1)):
            output_segments.append(datalist[start:start+output_segment_length])

        # We remove ___output_length__ items from the end of
        # input segments to allign them with the output segments.
        # We remove __input_length_segments___ items from __output_segments__
        input_segments = input_segments[:-output_segment_length]
        output_segments = output_segments[input_segment_length:]

        return(input_segments, output_segments)


    ####################################################################
    # Puts the number of period into the segment 
    # It also puts the number of subperiods  
    #
    def number_input_segments(self, datalist: list, number_of_subperiods=None, starting_subperiod=None)->list:

        resultlst = []

        if (number_of_subperiods!=None and starting_subperiod!=None): 
            ran = [r % number_of_subperiods for r in range(starting_subperiod, starting_subperiod + len(datalist))] 

        counter = 0
        for lst in datalist:
            if (number_of_subperiods!=None): 
                lst = [counter]+[ran[counter]]+lst
            else:
                lst = [counter]+lsta

            resultlst.append(lst)
            counter=counter+1

        print(resultlst)
        exit(0)
         

        return list(lst)
        

    @abstractmethod
    def generate_forecast(self) -> None:
        pass
