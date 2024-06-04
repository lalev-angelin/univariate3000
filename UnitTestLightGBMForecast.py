from LightGBMForecast import LightGBMForecast

forecast = LightGBMForecast(list(range(1, 50)),
            forecast_horizon=4,
            lookback=6, 
            num_leaves=2,
            max_depth=6, 
            )

forecast.generate_forecast()
