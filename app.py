from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests

# ----------------------------
# Load Trained ML Model
# ----------------------------
model = joblib.load("solar_monthly_model.pkl")

app = FastAPI(title="Solar Intelligence API")


# ----------------------------
# User Input Schema
# ----------------------------
class SolarInput(BaseModel):
    Latitude: float
    Longitude: float
    Month: int
    System_Size: float
    Tariff: float


@app.post("/predict")
def predict_solar(data: SolarInput):

    # Validate Month
    if data.Month < 1 or data.Month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    # ----------------------------
    # 1️⃣ Fetch Monthly Weather from NASA POWER
    # ----------------------------
    nasa_url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,RH2M,WS2M"
        f"&community=RE"
        f"&longitude={data.Longitude}"
        f"&latitude={data.Latitude}"
        f"&format=JSON"
    )

    try:
        response = requests.get(nasa_url, timeout=10)
        response.raise_for_status()
        nasa_data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA data")

    # Month mapping (IMPORTANT FIX)
    month_map = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
        5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
        9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }

    month_key = month_map[data.Month]

    try:
        climate = nasa_data["properties"]["parameter"]

        temp = climate["T2M"][month_key]
        humidity = climate["RH2M"][month_key]
        wind_speed = climate["WS2M"][month_key]

    except KeyError:
        raise HTTPException(status_code=500, detail="Weather data missing from NASA response")

    # ----------------------------
    # 2️⃣ Predict Solar Radiation (ML Model)
    # ----------------------------
    month_sin = np.sin(2 * np.pi * data.Month / 12)
    month_cos = np.cos(2 * np.pi * data.Month / 12)

    input_data = np.array([[
        temp,
        humidity,
        wind_speed,
        month_sin,
        month_cos
    ]])

    predicted_radiation = float(model.predict(input_data)[0])

    # ----------------------------
    # 3️⃣ Convert Radiation → Energy
    # ----------------------------
    PR = 0.8
    days_in_month = 30  # simplified

    monthly_energy = (
        data.System_Size *
        predicted_radiation *
        days_in_month *
        PR
    )

    yearly_energy = monthly_energy * 12

    # ----------------------------
    # 4️⃣ Financial Calculations
    # ----------------------------
    cost_per_kw = 60000
    total_cost = data.System_Size * cost_per_kw

    annual_savings = yearly_energy * data.Tariff
    payback_years = (
        total_cost / annual_savings
        if annual_savings != 0
        else None
    )

    # ----------------------------
    # 5️⃣ CO₂ Offset
    # ----------------------------
    co2_offset = yearly_energy * 0.82 / 1000

    # ----------------------------
    # Final Response
    # ----------------------------
    return {
        "Location": {
            "Latitude": data.Latitude,
            "Longitude": data.Longitude
        },
        "Weather_Data_Used": {
            "Temperature_C": round(temp, 2),
            "Humidity_%": round(humidity, 2),
            "Wind_Speed_mps": round(wind_speed, 2)
        },
        "Predicted_Solar_Radiation_kWh_m2_day": round(predicted_radiation, 3),
        "Yearly_Energy_kWh": round(yearly_energy, 2),
        "Annual_Savings_Rs": round(annual_savings, 2),
        "Payback_Years": round(payback_years, 2) if payback_years else None,
        "CO2_Offset_Tons_per_Year": round(co2_offset, 2)
    }
