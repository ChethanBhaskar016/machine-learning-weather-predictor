import time
from datetime import datetime

from fastapi.responses import JSONResponse

from models.humidity_classification import HumidityClassifier
from models.humidity_random_forest_regression import HumidityRegressor
from models.temperature_classification import TemperatureClassifier
from models.temperature_random_forest_regression import TemperatureRegressor
from models.data_processing import process_and_clean_data
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel


def train_all_models():
    print("Processing machine learning data...")
    process_and_clean_data()

    # Create instances of each model
    humidity_classifier = HumidityClassifier()
    humidity_regressor = HumidityRegressor()
    temperature_classifier = TemperatureClassifier()
    temperature_regressor = TemperatureRegressor()

    # Train Humidity Classifier
    print("Starting training for Humidity Classifier...")
    humidity_classifier.train()
    print("Humidity Classifier training completed.\n")

    # Train Humidity Regressor
    print("Starting training for Humidity Regressor...")
    humidity_regressor.train()
    print("Humidity Regressor training completed.\n")

    # Train Temperature Classifier
    print("Starting training for Temperature Classifier...")
    temperature_classifier.train()
    print("Temperature Classifier training completed.\n")

    # Train Temperature Regressor
    print("Starting training for Temperature Regressor...")
    temperature_regressor.train()
    print("Temperature Regressor training completed.\n")


# Initialize the FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # URL of React application
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request: {request.url} - Duration: {process_time} seconds")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "An error occurred"}
    )



async def check_month_middleware(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=404, detail="Please enter a month to predict.")
    if "month" not in body:
        raise HTTPException(status_code=404, detail="Please enter a month to predict.")
    try:
        month = int(body["month"])
    except:
        raise HTTPException(status_code=404, detail="Please enter a month number between January (1) and September (9)")

    if month < 1 or month > 9:
        raise HTTPException(status_code=404, detail="Please enter a month number between January (1) and September (9)")

    return


async def check_date_middleware(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=404, detail="Please enter a date in dd-mm format to predict.")

    if "date" not in body:
        raise HTTPException(status_code=404, detail="Please enter a date in dd-mm format to predict.")

    try:
        current_year = datetime.now().year
        date = datetime.strptime(f"{body['date']}-{current_year}", "%d-%m-%Y")
    except:
        raise HTTPException(status_code=404, detail="Please enter a date in dd-mm format which is within the range of "
                                                    "January 1 to September 22.")

    # Define the start and end dates for the range within the same year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime(current_year, 9, 22)

    # Check if the date is within the range
    if not start_date <= date <= end_date:
        raise HTTPException(status_code=404, detail="Please enter a date in dd-mm format which is within the range of "
                                                    "January 1 to September 22.")

    return


class Month(BaseModel):
    month: int


class Date(BaseModel):
    date: str


@app.post("/prediction/temperature-classification/monthly", dependencies=[Depends(check_month_middleware)])
async def temperature_classification(month: Month):
    month = int(month.month)
    temperature_classifier = TemperatureClassifier()
    predictions = temperature_classifier.predict(month=month)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


@app.post("/prediction/humidity-classification/monthly", dependencies=[Depends(check_month_middleware)])
async def humidity_classification(month: Month):
    month = int(month.month)
    humidity_classifier = HumidityClassifier()
    predictions = humidity_classifier.predict(month=month)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


@app.post("/prediction/temperature-regression/monthly", dependencies=[Depends(check_month_middleware)])
async def temperature_regression(month: Month):
    month = int(month.month)
    temperature_regressor = TemperatureRegressor()
    predictions = temperature_regressor.predict(month)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json, "month": month}


@app.post("/prediction/humidity-regression/monthly", dependencies=[Depends(check_month_middleware)])
async def humidity_regression(month: Month):
    month = int(month.month)
    humidity_regressor = HumidityRegressor()
    predictions = humidity_regressor.predict(month)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json, "month": month}


# @app.post("/prediction/temperature-regression/day-average")
# async def temperature_regression(date: Date):
#     temperature_regressor = TemperatureRegressor()
#     predictions = temperature_regressor.predict_day(date.date, avg=True)
#     predictions_json = predictions.to_dict(orient="records")
#     return {"Temperature Predictions": predictions_json}
#
#
# @app.post("/prediction/humidity-regression/day-average")
# async def humidity_regression(date: Date):
#     humidity_regressor = HumidityRegressor()
#     predictions = humidity_regressor.predict_day(date.date, avg=True)
#     predictions_json = predictions.to_dict(orient="records")
#     return {"Humidity Predictions": predictions_json}


@app.post("/prediction/temperature-regression/day-hourly", dependencies=[Depends(check_date_middleware)])
async def temperature_regression(date: Date):
    temperature_regressor = TemperatureRegressor()
    predictions = temperature_regressor.predict_day(date.date)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


@app.post("/prediction/humidity-regression/day-hourly", dependencies=[Depends(check_date_middleware)])
async def humidity_regression(date: Date):
    humidity_regressor = HumidityRegressor()
    predictions = humidity_regressor.predict_day(date.date)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


@app.post("/prediction/humidity-classification/day-hourly", dependencies=[Depends(check_date_middleware)])
async def humidity_classification(date: Date):
    humidity_classifier = HumidityClassifier()
    predictions = humidity_classifier.predict_day(date.date)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


@app.post("/prediction/temperature-classification/day-hourly", dependencies=[Depends(check_date_middleware)])
async def temperature_classification(date: Date):
    temperature_classifier = TemperatureClassifier()
    predictions = temperature_classifier.predict_day(date.date)
    predictions_json = predictions.to_dict(orient="records")
    return {"Predictions": predictions_json}


if __name__ == "__main__":
    print("Training all models...\n")
    train_all_models()  # Train the models first
    print("All models have been successfully trained!")

    # Start the FastAPI app after the models are trained
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

