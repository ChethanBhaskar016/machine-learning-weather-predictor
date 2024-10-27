import joblib
import pandas as pd
import re
import os
from sklearn.preprocessing import LabelEncoder


def process_and_clean_data():  # returns the processed and cleaned weather data
    # Getting the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the data files using absolute paths
    microclimate_sensors_data_file = os.path.join(current_dir, "microclimate-sensors-data.csv")
    argyle_square_sensor_data_file = os.path.join(current_dir, "meshed-sensor-type-1.csv")

    # Loading the weather data, Argyle Square data
    microclimate_sensors_data = pd.read_csv(microclimate_sensors_data_file)
    argyle_square_sensor_data = pd.read_csv(argyle_square_sensor_data_file)

    # Selecting relevant columns for processing
    microclimate_sensors_data = microclimate_sensors_data[
        ["received_at", "sensorlocation", "airtemperature", "relativehumidity", 'atmosphericpressure']]
    argyle_square_sensor_data = argyle_square_sensor_data[
        ["time", "relativehumidity", "airtemp", "atmosphericpressure"]]

    # Removing any escape characters in the textual columns.
    def clean_text(location):
        if isinstance(location, str):  # Only apply cleaning if the value is a string
            cleaned_location = re.sub(r'[\n\t\r]', ' ', location)
            cleaned_location = re.sub(r'\s+', ' ', cleaned_location).strip()
            return cleaned_location
        return location

    microclimate_sensors_data['sensorlocation'] = microclimate_sensors_data['sensorlocation'].apply(clean_text)
    microclimate_sensors_data['received_at'] = microclimate_sensors_data['received_at'].apply(clean_text)
    argyle_square_sensor_data['time'] = argyle_square_sensor_data['time'].apply(clean_text)

    # Converting received_at and Date columns to datetime
    microclimate_sensors_data['received_at'] = pd.to_datetime(microclimate_sensors_data['received_at'], errors='coerce',
                                                              utc=True)
    argyle_square_sensor_data['time'] = pd.to_datetime(argyle_square_sensor_data['time'], errors='coerce', utc=True)

    # Extracting hour, day, and month from the datetime columns
    microclimate_sensors_data['hour'] = microclimate_sensors_data['received_at'].dt.hour
    microclimate_sensors_data['month'] = microclimate_sensors_data['received_at'].dt.month
    microclimate_sensors_data['day'] = microclimate_sensors_data['received_at'].dt.day
    argyle_square_sensor_data['month'] = argyle_square_sensor_data['time'].dt.month
    argyle_square_sensor_data['day'] = argyle_square_sensor_data['time'].dt.day
    argyle_square_sensor_data['hour'] = argyle_square_sensor_data['time'].dt.hour

    # Formatting dates in both dataframes to 'dd-mm-yy'
    microclimate_sensors_data['received_at'] = microclimate_sensors_data['received_at'].dt.strftime('%d-%m-%y')
    argyle_square_sensor_data['time'] = argyle_square_sensor_data['time'].dt.strftime('%d-%m-%y')

    # Renaming the columns in argyle square data to match the names of microclimate_sensors_data columns
    argyle_square_sensor_data.rename(columns={'time': 'Date', 'airtemp': 'airtemperature'}, inplace=True)
    argyle_square_sensor_data['sensorlocation'] = 'Argyle Square'

    # Renaming received_at to Date in microclimate_sensors_data
    microclimate_sensors_data.rename(columns={'received_at': 'Date'}, inplace=True)

    # Concatenating microclimate_sensors_data and argyle square data
    microclimate_sensors_data = pd.concat([microclimate_sensors_data, argyle_square_sensor_data], ignore_index=True)

    # Converting numerical columns to numeric and handle missing values
    # Mean imputing into the null values, removing any non numeric data, and rounding the numeric data to 1 decimal point.

    for column in ['airtemperature', 'relativehumidity', 'atmosphericpressure']:
        microclimate_sensors_data[column] = pd.to_numeric(microclimate_sensors_data[column], errors='coerce')
        microclimate_sensors_data[column] = microclimate_sensors_data[column].fillna(
            microclimate_sensors_data[column].mean())
        microclimate_sensors_data.loc[:, column] = microclimate_sensors_data[column].round(1)

    # Outlier detection using IQR method
    weather_data_numeric_columns = ['airtemperature', 'relativehumidity', 'atmosphericpressure']
    Q1 = microclimate_sensors_data[weather_data_numeric_columns].quantile(0.25)
    Q3 = microclimate_sensors_data[weather_data_numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for column in weather_data_numeric_columns:
        microclimate_sensors_data = microclimate_sensors_data[
            (microclimate_sensors_data[column] >= lower_bound[column]) & (
                        microclimate_sensors_data[column] <= upper_bound[column])]

    # Removing any unexpected sensor location in the data
    unique_sensor_locations = {'Batman Park', 'CH1 rooftop',
                               'Tram Stop 7C - Melbourne Tennis Centre Precinct - Rod Laver Arena',
                               "SkyFarm (Jeff's Shed). Rooftop - Melbourne Conference & Exhibition Centre (MCEC)",
                               'Royal Park Asset ID: COM2707',
                               'Tram Stop 7B - Melbourne Tennis Centre Precinct - Rod Laver Arena',
                               '101 Collins St L11 Rooftop', 'Birrarung Marr Park - Pole 1131',
                               'Enterprize Park - Pole ID: COM1667',
                               'Swanston St - Tram Stop 13 adjacent Federation Sq & Flinders St Station',
                               'Argyle Square'}

    microclimate_sensors_data = microclimate_sensors_data[
        microclimate_sensors_data['sensorlocation'].isin(unique_sensor_locations)]

    # Mapping the month to Melbourne's seasons for machine learning tasks
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        elif month in [9, 10, 11]:
            return 'Spring'

    # Adding the 'season' column
    microclimate_sensors_data['season'] = microclimate_sensors_data['month'].apply(get_season)

    # Encoding the sensor locations and seasons into numerical values for machine learning tasks
    label_encoder = LabelEncoder()
    microclimate_sensors_data['sensor-location-encoded'] = label_encoder.fit_transform(
        microclimate_sensors_data['sensorlocation'])
    microclimate_sensors_data['season-encoded'] = label_encoder.fit_transform(microclimate_sensors_data['season'])

    # Dropping any null values left
    microclimate_sensors_data = microclimate_sensors_data.dropna()
    microclimate_sensors_data.reset_index(drop=True, inplace=True)

    processed_data_file = os.path.join(current_dir, 'processed_weather_data.pkl')
    joblib.dump(microclimate_sensors_data, processed_data_file)

    return microclimate_sensors_data


# Execute the function
process_and_clean_data()
