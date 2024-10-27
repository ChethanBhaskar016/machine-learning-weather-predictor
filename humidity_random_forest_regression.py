

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os
import pandas as pd


class HumidityRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.weather_data_file = 'processed_weather_data.pkl'
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'processed_weather_data.pkl')
        self.weather_data = joblib.load(file_path)

    def train(self):
        # Select features (X) and target (y) from the dataset
        X = self.weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]
        y = self.weather_data['relativehumidity']

        # Initialize and train the Random Forest Regression model
        self.model.fit(X, y)

        # Make predictions on the test set
        y_pred = self.model.predict(X)

        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'humidity_regression_model.pkl')
        joblib.dump(self.model, file_path)

        # Calculate evaluation metrics
        mse = mean_squared_error(y, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mae = mean_absolute_error(y, y_pred)  # Mean Absolute Error
        r2 = r2_score(y, y_pred)  # R^2 Score (coefficient of determination)

        # Print the evaluation metrics
        print('Mean Squared Error (MSE): %.2f' % mse)
        print('Root Mean Squared Error (RMSE): %.2f' % rmse)
        print('Mean Absolute Error (MAE): %.2f' % mae)
        print('R^2 Score: %.2f' % r2)

    def predict(self, month=None):
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'], format='%d-%m-%y', errors='coerce')

        # Filter the data for the year 2024
        weather_data = self.weather_data[self.weather_data['Date'].dt.year == 2024]
        if month:
            weather_data = weather_data[weather_data["month"] == month]
        # Selecting the features (X) for the model (without splitting the dataset)
        X = weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]

        # Assuming there is a 'date' column in your dataset, we store it separately
        dates = weather_data['Date']

        # Load the trained model
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'humidity_regression_model.pkl')
        model = joblib.load(file_path)

        # Make predictions for all entries in the dataset
        predictions = model.predict(X)

        # Create a DataFrame to store dates and predictions
        result = pd.DataFrame({
            'Date': dates,
            'Prediction': predictions
        })

        result = result.groupby('Date', as_index=False)['Prediction'].mean()

        # Format the Date column to 'dd-mm-yy'
        result['Date'] = result['Date'].dt.strftime('%d')

        # Return the DataFrame with one entry per day (average predicted temperature)
        return result

    def predict_day(self, date="01-01", avg=False):
        date = date.split("-")
        month = int(date[1])
        day = int(date[0])

        # Ensure that the Date column is in datetime format
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'], format='%d-%m-%y', errors='coerce')

        # Filter the data for the year 2024
        weather_data = self.weather_data[self.weather_data['Date'].dt.year == 2024]

        # Filter the data for a specific month and day provided
        weather_data = weather_data[(weather_data["month"] == month) & (weather_data["Date"].dt.day == day)]

        # Selecting the features (X) for the model
        X = weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]

        # Store the Date column
        hours = weather_data['hour']

        # Load the trained model
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'humidity_regression_model.pkl')
        model = joblib.load(file_path)

        # Make predictions for all entries in the dataset
        predictions = model.predict(X)

        # Create a DataFrame to store dates and predictions
        result = pd.DataFrame({
            'hour': hours,
            'Prediction': predictions
        })

        # Group by 'Date' and calculate the average predicted temperature for each day
        result = result.groupby('hour', as_index=False)['Prediction'].mean()

        # Return the DataFrame with one entry per day (average predicted temperature)
        return result



