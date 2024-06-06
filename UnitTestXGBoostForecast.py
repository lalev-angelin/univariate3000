from XGBoostForecast import XGBoostForecast

forecast = XGBoostForecast(list(range(1, 50)),
            forecast_horizon=4,
            lookback=6, 
            num_leaves=2,
            max_depth=6, 
            min_data_in_leaf=2,
            min_data_in_bin=2
            )

forecast.generate_forecast()
print(forecast.as_list())
