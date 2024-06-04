from GBDTForecast import GBDTForecast
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

class LightGBMForecast(GBDTForecast):

    def __init__(self, 
            timeseries: list, 
            dates: list = None, 
            forecast_horizon : int = 1, 
            lookback : int = 1,
            number_of_subperiods : int = None,
            starting_subperiod : int = 1, 
            boosting_type: str = 'gbdt',
            num_leaves: int = 31, 
            max_depth: int = -1, 
            min_data_in_leaf: int = None,
            learning_rate: float = 0.1,
            n_estimators: int = 100, 
            subsample_for_bin: int = 200000, 
            class_weight : dict = None, 
            min_split_gain: float = 0, 
            min_child_weight: float = 1e-3, 
            min_child_samples: int =  20, 
            subsample: float = 1.0,
            subsample_freq: int = 0, 
            reg_alpha: float = 0.0,
            reg_lambda: float = 0.0,
            random_state: int = None, 
            n_jobs: int = None,
            **kwargs):
        
        super().__init__(timeseries, 
                     dates=dates, 
                     forecast_horizon = forecast_horizon, 
                     lookback = lookback, 
                     number_of_subperiods = number_of_subperiods,
                     starting_subperiod = starting_subperiod)

        self._boosting_type = boosting_type
        self._num_leaves = num_leaves
        self._max_depth = max_depth
        self._min_data_in_leaf=min_data_in_leaf
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators 
        self._subsample_for_bin = subsample_for_bin
        self._class_weight = class_weight
        self._min_split_gain = min_split_gain 
        self._min_child_weight = min_child_weight 
        self._min_child_samples = min_child_samples 
        self._subsample = subsample
        self._subsample_freq = subsample_freq 
        self._reg_alpha = reg_alpha 
        self._reg_lambda = reg_lambda 
        self._random_state = random_state
        self._n_jobs = n_jobs 
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
                boosting_type=self._boosting_type,
                num_leaves=self._num_leaves,
                max_depth=self._max_depth,
                min_data_in_leaf=self._min_data_in_leaf,
                learning_rate=self._learning_rate,
                n_estimators=self._n_estimators,
                subsample_for_bin=self._subsample_for_bin,
                class_weight=self._class_weight,
                min_split_gain=self._min_split_gain,
                min_child_weight=self._min_child_weight,
                min_child_samples=self._min_child_samples, 
                subsample=self._subsample,
                subsample_freq=self._subsample_freq,
                reg_alpha=self._reg_alpha,
                reg_lambda=self._reg_lambda,
                random_state=self._random_state,
                n_jobs=self._n_jobs,
                **self._kwargs)
        
        multi_target_regressor = MultiOutputRegressor(regressor)

        multi_target_regressor.fit(trainX, trainY)

        forecast = multi_target_regressor.predict(input_segments) 

        
