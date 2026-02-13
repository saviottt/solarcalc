from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import requests

# ----------------------------
# Load Kerala Location-Aware Monthly Model
# ----------------------------
model = joblib.load("kerala_monthly_location_model.pkl")

app = FastAPI(title="Kerala Smart Solar Intelligence API")

# ----------------------------
# Enable CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# User Input Schema
# ----------------------------
class SolarInput(BaseModel):
    Latitude: float
    Longitude: float
    Month: int
    System_Size: float
    Tariff: float                  # Buying tariff
    Monthly_Consumption: float     # kWh per month
    EV_Daily_KM: float             # Daily km driven
    EV_Efficiency: float           # km per kWh
    Export_Tariff: float           # Selling price to KSEB


@app.post("/predict")
def predict_solar(data: SolarInput):

    if data.Month < 1 or data.Month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    # ----------------------------
    # Fetch NASA Climate
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
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA data")

    month_map = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
        5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
        9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }

    climate = nasa_data["properties"]["parameter"]
    month_key = month_map[data.Month]

    temp = climate["T2M"][month_key]
    humidity = climate["RH2M"][month_key]
    wind_speed = climate["WS2M"][month_key]

    # ----------------------------
    # Predict Solar Radiation
    # ----------------------------
    month_sin = np.sin(2 * np.pi * data.Month / 12)
    month_cos = np.cos(2 * np.pi * data.Month / 12)

    input_data = np.array([[
        data.Latitude,
        data.Longitude,
        temp,
        humidity,
        wind_speed,
        month_sin,
        month_cos
    ]])

    predicted_radiation = float(model.predict(input_data)[0])

    # ----------------------------
    # Solar Energy Calculation
    # ----------------------------
    PR = 0.8
    days_in_month = 30

    monthly_energy = data.System_Size * predicted_radiation * days_in_month * PR
    yearly_energy = monthly_energy * 12

    # ----------------------------
    # Home Usage
    # ----------------------------
    yearly_home_consumption = data.Monthly_Consumption * 12

    # ----------------------------
    # EV Usage
    # ----------------------------
    ev_daily_energy = data.EV_Daily_KM / data.EV_Efficiency if data.EV_Efficiency != 0 else 0
    ev_yearly_energy = ev_daily_energy * 365

    # ----------------------------
    # Total Usage
    # ----------------------------
    total_yearly_usage = yearly_home_consumption + ev_yearly_energy

    surplus_energy = yearly_energy - total_yearly_usage

    # ----------------------------
    # Financial Logic
    # ----------------------------
    cost_per_kw = 60000
    total_cost = data.System_Size * cost_per_kw

    if surplus_energy >= 0:
        # Fully covered + export
        savings_from_usage = total_yearly_usage * data.Tariff
        export_income = surplus_energy * data.Export_Tariff
        grid_purchase_cost = 0
    else:
        # Not enough solar
        savings_from_usage = yearly_energy * data.Tariff
        export_income = 0
        grid_purchase_cost = abs(surplus_energy) * data.Tariff

    annual_net_benefit = savings_from_usage + export_income - grid_purchase_cost

    payback_years = total_cost / annual_net_benefit if annual_net_benefit > 0 else None

    # ----------------------------
    # CO2 Offset
    # ----------------------------
    co2_offset = yearly_energy * 0.82 / 1000

    # ----------------------------
    # 25-Year Projection
    # ----------------------------
    degradation_rate = 0.005
    years = 25

    total_savings_25 = 0

    for year in range(years):
        degraded_energy = yearly_energy * ((1 - degradation_rate) ** year)

        degraded_surplus = degraded_energy - total_yearly_usage

        if degraded_surplus >= 0:
            yearly_savings = (total_yearly_usage * data.Tariff) + \
                             (degraded_surplus * data.Export_Tariff)
        else:
            yearly_savings = (degraded_energy * data.Tariff) - \
                             (abs(degraded_surplus) * data.Tariff)

        total_savings_25 += yearly_savings

    net_profit_25 = total_savings_25 - total_cost

    # ----------------------------
    # Final Response
    # ----------------------------
    return {
        "Solar_Generation_kWh_per_Year": round(yearly_energy, 2),
        "Home_Consumption_kWh_per_Year": round(yearly_home_consumption, 2),
        "EV_Consumption_kWh_per_Year": round(ev_yearly_energy, 2),
        "Total_Usage_kWh_per_Year": round(total_yearly_usage, 2),
        "Surplus_or_Deficit_kWh": round(surplus_energy, 2),

        "Export_Income_Rs": round(export_income, 2),
        "Grid_Purchase_Cost_Rs": round(grid_purchase_cost, 2),

        "Annual_Net_Benefit_Rs": round(annual_net_benefit, 2),
        "Payback_Years": round(payback_years, 2) if payback_years else None,

        "CO2_Offset_Tons_per_Year": round(co2_offset, 2),

        "25_Year_Total_Savings_Rs": round(total_savings_25, 2),
        "25_Year_Net_Profit_Rs": round(net_profit_25, 2)
    }
