from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import requests

# ----------------------------
# Load Model
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
# Constants
# ----------------------------
PR = 0.8
COST_PER_KW = 60000
DEGRADATION_RATE = 0.005
YEARS_PROJECTION = 25
EXPORT_TARIFF = 3.15

MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31
}

# ----------------------------
# Input Schema
# ----------------------------
class SolarInput(BaseModel):
    Latitude: float
    Longitude: float
    System_Size: float
    Monthly_Consumption: float
    EV_Daily_KM: Optional[float] = 0
    EV_Efficiency: Optional[float] = 0
    Tilt: Optional[float] = None
    NOCT: Optional[float] = 45

# ----------------------------
# KSEB Slab Billing
# ----------------------------
def calculate_kseb_domestic_bill(units: float) -> float:
    slabs = [
        (50, 3.15),
        (50, 3.70),
        (100, 4.80),
        (100, 6.40),
        (float("inf"), 7.50),
    ]

    remaining = units
    cost = 0.0

    for slab_units, rate in slabs:
        if remaining <= 0:
            break
        used = min(remaining, slab_units)
        cost += used * rate
        remaining -= used

    return cost

# ----------------------------
# NASA Climate Fetch
# ----------------------------
def fetch_nasa_climate(lat: float, lon: float):
    nasa_url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,RH2M,WS2M"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&format=JSON"
    )

    try:
        response = requests.get(nasa_url, timeout=10)
        response.raise_for_status()
        return response.json()["properties"]["parameter"]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA data")

# ----------------------------
# Solar Physics Helpers
# ----------------------------
def calculate_optimal_tilt(latitude: float) -> float:
    return abs(0.76 * latitude + 3.1)

def calculate_cell_temperature(ambient_temp, irradiance_kwh_m2_day, noct):
    irradiance_w_m2 = irradiance_kwh_m2_day * 1000 / 24
    return ambient_temp + ((noct - 20) / 800) * irradiance_w_m2

def temperature_derating_from_cell(cell_temp):
    effect = 1 + (-0.004 * (cell_temp - 25))
    return max(0.75, min(effect, 1.05))

def apply_tilt_correction(ghi, latitude, tilt):
    latitude_rad = np.radians(latitude)
    tilt_rad = np.radians(tilt)
    tilt_factor = np.cos(latitude_rad - tilt_rad) / np.cos(latitude_rad)
    tilt_factor = max(0.85, min(1.15, tilt_factor))
    return ghi * tilt_factor

# ----------------------------
# PM Surya Ghar Subsidy
# ----------------------------
def calculate_pm_surya_ghar_subsidy(system_size_kw: float) -> float:
    if system_size_kw <= 2:
        return system_size_kw * 30000
    elif system_size_kw <= 3:
        return 60000 + (system_size_kw - 2) * 18000
    else:
        return 78000

# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
def predict_solar(data: SolarInput):

    optimal_tilt = calculate_optimal_tilt(data.Latitude)
    tilt_used = data.Tilt if data.Tilt is not None else optimal_tilt

    climate = fetch_nasa_climate(data.Latitude, data.Longitude)
    yearly_energy = 0

    # Monthly Simulation
    for month in range(1, 13):

        month_key = MONTH_MAP[month]
        temp = climate["T2M"][month_key]

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        input_data = np.array([[
            data.Latitude,
            data.Longitude,
            temp,
            climate["RH2M"][month_key],
            climate["WS2M"][month_key],
            month_sin,
            month_cos
        ]])

        predicted_ghi = float(model.predict(input_data)[0])

        corrected_irradiance = apply_tilt_correction(
            predicted_ghi,
            data.Latitude,
            tilt_used
        )

        cell_temp = calculate_cell_temperature(
            temp,
            corrected_irradiance,
            data.NOCT
        )

        temp_effect = temperature_derating_from_cell(cell_temp)

        monthly_energy = (
            data.System_Size *
            corrected_irradiance *
            DAYS_IN_MONTH[month] *
            PR *
            temp_effect
        )

        yearly_energy += monthly_energy

    # ----------------------------
    # Consumption
    # ----------------------------
    yearly_home_consumption = data.Monthly_Consumption * 12

    if data.EV_Daily_KM and data.EV_Efficiency and data.EV_Efficiency > 0:
        ev_yearly_energy = (data.EV_Daily_KM / data.EV_Efficiency) * 365
    else:
        ev_yearly_energy = 0

    total_yearly_usage = yearly_home_consumption + ev_yearly_energy

    surplus_energy = yearly_energy - total_yearly_usage
    grid_import_units = max(0, total_yearly_usage - yearly_energy)

    # ----------------------------
    # KSEB Billing
    # ----------------------------
    yearly_bill_without_solar = calculate_kseb_domestic_bill(total_yearly_usage)
    yearly_bill_with_solar = calculate_kseb_domestic_bill(grid_import_units)

    export_income = max(0, surplus_energy) * EXPORT_TARIFF
    annual_net_benefit = (
        yearly_bill_without_solar -
        yearly_bill_with_solar +
        export_income
    )

    # ----------------------------
    # Cost + Subsidy
    # ----------------------------
    gross_cost = data.System_Size * COST_PER_KW
    subsidy = calculate_pm_surya_ghar_subsidy(data.System_Size)
    net_cost = gross_cost - subsidy

    payback_years = (
        net_cost / annual_net_benefit
        if annual_net_benefit > 0 else None
    )

    # ----------------------------
    # CO2
    # ----------------------------
    co2_offset = yearly_energy * 0.82 / 1000

    # ----------------------------
    # 25-Year Projection
    # ----------------------------
    total_savings_25 = 0

    for year in range(YEARS_PROJECTION):

        degraded_energy = yearly_energy * ((1 - DEGRADATION_RATE) ** year)
        degraded_import = max(0, total_yearly_usage - degraded_energy)
        bill_with_solar = calculate_kseb_domestic_bill(degraded_import)

        yearly_savings = yearly_bill_without_solar - bill_with_solar
        degraded_surplus = max(0, degraded_energy - total_yearly_usage)
        yearly_savings += degraded_surplus * EXPORT_TARIFF

        total_savings_25 += yearly_savings

    net_profit_25 = total_savings_25 - net_cost

    # ----------------------------
    # Response
    # ----------------------------
    return {
        "Solar_Generation_kWh_per_Year": round(yearly_energy, 2),
        "Total_Usage_kWh_per_Year": round(total_yearly_usage, 2),

        "Optimal_Tilt_Suggested_Degrees": round(optimal_tilt, 2),
        "Panel_Tilt_Used_Degrees": round(tilt_used, 2),

        "Gross_System_Cost_Rs": round(gross_cost, 2),
        "PM_Surya_Ghar_Subsidy_Rs": round(subsidy, 2),
        "Net_System_Cost_Rs": round(net_cost, 2),

        "Annual_Bill_Without_Solar_Rs": round(yearly_bill_without_solar, 2),
        "Annual_Bill_With_Solar_Rs": round(yearly_bill_with_solar, 2),
        "Annual_Net_Benefit_Rs": round(annual_net_benefit, 2),
        "Payback_Years": round(payback_years, 2) if payback_years else None,

        "CO2_Offset_Tons_per_Year": round(co2_offset, 2),

        "25_Year_Total_Savings_Rs": round(total_savings_25, 2),
        "25_Year_Net_Profit_Rs": round(net_profit_25, 2)
    }
