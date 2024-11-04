<H1 ALIGN =CENTER> EX.NO.09 --  A project on Time Series Analysis on Weather Forecasting using ARIMA model ...</H1>

### Date: 27-04-2024

### AIM :

To Create a project on Time series analysis on weather forecasting using ARIMA model in python and compare with other models.

### ALGORITHM :

#### Step 1 : 

Explore the dataset of weather.

#### Step 2 : 

Check for stationarity of time series time series plot :

   ACF plot and PACF plot
   
   ADF test
   
   Transform to stationary: differencing
   
#### Step 3 : 

Determine ARIMA models parameters p, q.

#### Step 4 : 

Fit the ARIMA model.

#### Step 5 :

Make time series predictions.

#### Step 6 : 

Auto-fit the ARIMA model.

#### Step 7 : 

Evaluate model predictions.

### PROGRAM :

#### Import the neccessary packages :

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```

#### Load the dataset :

```python
data = pd.read_csv("/content/seattle-weather.csv")
```

#### Convert 'Date' column to datetime format :

```python
data['date'] = pd.to_datetime(data['date'])
```

#### Set 'Date' column as index :

```python
data.set_index('date', inplace=True)
```

#### Arima Model :

```python

def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

arima_model(data, 'temp_max', order=(5,1,0))

```

### OUTPUT :

![img1](https://github.com/anto-richard/TSA_EXP9/assets/93427534/8fcae10b-39cf-458f-a5d2-9b3abb952e05)

### RESULT :

Thus, the program successfully executted based on the ARIMA model using python.
