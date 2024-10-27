from sklearn.ensemble import RandomForestClassifier  # For Random Forest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


class HumidityClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.weather_data_file = 'processed_weather_data.pkl'
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'processed_weather_data.pkl')
        self.weather_data = joblib.load(file_path)

    def train(self):
        # Selecting the features (X) and target (y) for the model
        X = self.weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]

        # Discretizing the 'relativehumidity' column into categories (low, medium, high)
        humidity_bins = [0, 30, 60, 80, 100]
        humidity_labels = ['very low', 'low', 'medium', 'high']
        self.weather_data['humidity_category'] = pd.cut(self.weather_data['relativehumidity'], bins=humidity_bins,
                                                        labels=humidity_labels)

        # The target variable is 'humidity_category'
        y = self.weather_data['humidity_category']

        # Standardizing the features (scaling to have mean=0 and variance=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)  # Fit and transform the training data

        # Save the scaler for later use in prediction
        scaler_file_path = os.path.join(os.path.dirname(__file__), 'humidity_scaler.pkl')
        joblib.dump(scaler, scaler_file_path)

        # Train the Random Forest model on the scaled training data
        self.model.fit(X_train_scaled, y)

        # Save the trained model
        model_file_path = os.path.join(os.path.dirname(__file__), 'humidity_classification_model.pkl')
        joblib.dump(self.model, model_file_path)

        # Evaluating the model's performance
        y_pred = self.model.predict(X_train_scaled)
        print(f"Humidity classification Evaluation:")
        print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
        print(f"Precision: {precision_score(y, y_pred, average='weighted'):.2f}")
        print(f"Recall: {recall_score(y, y_pred, average='weighted'):.2f}")
        print(f"F1-Score: {f1_score(y, y_pred, average='weighted'):.2f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        print('-' * 50)

    def predict(self, month=None):
        # Selecting the features (X) for the model

        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'], format='%d-%m-%y', errors='coerce')

        # Filter the data for the specific month and day
        if month:
            month = int(month)
            weather_data = self.weather_data[(self.weather_data["month"] == month)]
            weather_data = weather_data[weather_data['Date'].dt.year == 2024]
        else:
            weather_data = self.weather_data[self.weather_data['Date'].dt.year == 2024]

        X = weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]
        dates = weather_data['Date']

        # Load the saved model
        current_dir = os.path.dirname(__file__)
        model_file_path = os.path.join(current_dir, 'humidity_classification_model.pkl')
        model = joblib.load(model_file_path)

        # Load the saved scaler and scale the features
        scaler_file_path = os.path.join(current_dir, 'humidity_scaler.pkl')
        scaler = joblib.load(scaler_file_path)
        X_scaled = scaler.transform(X)

        # Make predictions for all entries in the dataset
        predictions = model.predict(X_scaled)

        # Create a DataFrame to store dates and predictions
        result = pd.DataFrame({
            'Date': dates,
            'Prediction': predictions
        })

        result_grouped = result.groupby('Date', as_index=False).agg(lambda x: pd.Series.mode(x)[0])

        result_grouped['Date'] = result_grouped['Date'].dt.strftime('%d-%m-%y')
        # Return the DataFrame
        return result_grouped

    def predict_day(self, date="01-01"):
        # Split and extract the month and day from the input date
        date = date.split("-")
        month = int(date[1])
        day = int(date[0])

        # Ensure that the Date column is in datetime format
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'], format='%d-%m-%y', errors='coerce')

        # Filter the data for the year 2024
        weather_data = self.weather_data[self.weather_data['Date'].dt.year == 2024]

        # Filter the data for the specific month and day
        weather_data = weather_data[(weather_data["month"] == month) & (weather_data["Date"].dt.day == day)]

        # Selecting the features (X) for the model
        X = weather_data[
            ['airtemperature', 'day', 'hour', 'sensor-location-encoded', 'atmosphericpressure', 'season-encoded']]

        # Store the hour column
        hours = weather_data['hour']

        # Load the trained model
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'humidity_classification_model.pkl')
        model = joblib.load(file_path)

        # Load the scaler
        scaler_file_path = os.path.join(current_dir, 'humidity_scaler.pkl')
        scaler = joblib.load(scaler_file_path)

        # Scale the input features
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Create a DataFrame to store hours and predictions
        result = pd.DataFrame({
            'hour': hours,
            'Prediction': predictions
        })

        # Group by 'hour' and calculate the most frequent 'Classified Humidity' value for each hour
        result_grouped = result.groupby('hour', as_index=False).agg(lambda x: pd.Series.mode(x)[0])

        # Return the DataFrame with one entry per hour (most frequent classified humidity)
        return result_grouped
