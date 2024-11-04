<H1 ALIGN =CENTER> EX.NO.09 --  A project on Time Series Analysis on Weather Forecasting using ARIMA model ...</H1>

### Date: 

### AIM :

To Create a project on Time series analysis on weather forecasting using ARIMA model in python and compare with other models.

### ALGORITHM :

1. Explore the dataset of weather
2. Check for stationarity of time series time series plot ACF plot and PACF plot ADF test Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions


### PROGRAM :
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the Nvidia dataset from the specified file path
data_path = "C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv"
data = pd.read_csv(data_path)

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filter data from 2016 to 2024
data = data.loc[(data.index >= '2016-01-01') & (data.index <= '2024-12-31')]

# Display the first few rows of the dataset to confirm the date filter
print(data.head())

def arima_model(data, target_variable, order):
    # Split data into training and testing sets (80% training, 20% testing)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model on the training data
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast for the length of the test data
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the results
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

# Example usage with 'Close' as the target variable and ARIMA order (5, 1, 0)
arima_model(data, 'Close', order=(5, 1, 0))

```

### OUTPUT :
![image](https://github.com/user-attachments/assets/7fe2040a-b8ce-4762-a1b3-156df344460f)

### RESULT :

Thus, the program successfully executted based on the ARIMA model using python.
