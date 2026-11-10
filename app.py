from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import pandas as pd
import joblib
import os
import requests
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "floodsense-local-dev-key")

# ==========================================
# Paths and Model Loading
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "floods.save")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")

DATA_FOLDER = os.path.join(BASE_DIR, "data")
HISTORY_FILE = os.path.join(DATA_FOLDER, "prediction_history.csv")
MONITOR_HISTORY_FILE = os.path.join(DATA_FOLDER, "monitor_history.csv")

os.makedirs(DATA_FOLDER, exist_ok=True)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("=" * 60)
print("FloodSense AI - Model Loaded Successfully")
print("Model Type:", type(model))
print("Model Features:", getattr(model, "n_features_in_", "Unknown"))
print("Scaler Features:", getattr(scaler, "n_features_in_", "Unknown"))
print("=" * 60)

PREDICTION_COLUMNS = ["Date", "Prediction", "Confidence"]

MONITOR_COLUMNS = [
    "Timestamp",
    "Location",
    "Latitude",
    "Longitude",
    "Rain next 24h (mm)",
    "Max rain probability (%)",
    "Weather concern"
]


def initialize_csv(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


initialize_csv(HISTORY_FILE, PREDICTION_COLUMNS)
initialize_csv(MONITOR_HISTORY_FILE, MONITOR_COLUMNS)


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# About
# ==========================================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================
# Flood Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_names = [
            "MonsoonIntensity",
            "DrainageSystems",
            "RiverManagement",
            "Deforestation",
            "Urbanization"
        ]

        values = []

        for name in feature_names:
            value = float(request.form[name])

            if not np.isfinite(value) or not 0 <= value <= 10:
                return (
                    "All five input values must be between 0 and 10.",
                    400
                )

            values.append(value)

        input_data = pd.DataFrame([values], columns=feature_names)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        confidence = round(float(np.max(probabilities)) * 100, 2)

        if confidence >= 90:
            confidence_level = "Very High"
            confidence_color = "success"
        elif confidence >= 70:
            confidence_level = "High"
            confidence_color = "primary"
        elif confidence >= 50:
            confidence_level = "Moderate"
            confidence_color = "warning"
        else:
            confidence_level = "Low"
            confidence_color = "danger"

        if prediction == 1:
            prediction_text = "HIGH FLOOD RISK"
            risk = "High"
            color = "danger"

            recommendation = [
                "Check official flood and weather advisories.",
                "Prepare an emergency plan if flooding is possible.",
                "Avoid walking or driving through floodwater.",
                "Monitor local rainfall and water-level updates.",
                "Follow instructions from local authorities."
            ]
        else:
            prediction_text = "LOW FLOOD RISK"
            risk = "Low"
            color = "success"

            recommendation = [
                "Continue monitoring official weather updates.",
                "Maintain drainage systems where possible.",
                "Stay alert during periods of heavy rainfall.",
                "Check local advisories if conditions change.",
                "Follow official safety instructions."
            ]

        # Save prediction history
        history_data = pd.read_csv(HISTORY_FILE)

        history_data.loc[len(history_data)] = [
            datetime.now().strftime("%d-%m-%Y %H:%M"),
            prediction_text,
            confidence
        ]

        history_data.to_csv(HISTORY_FILE, index=False)

        return render_template(
            "result.html",
            prediction=prediction_text,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_color=confidence_color,
            risk=risk,
            color=color,
            recommendation=recommendation
        )

    except (ValueError, KeyError):
        return "Please enter valid numeric values for all five inputs.", 400

    except Exception:
        app.logger.exception("Prediction failed")
        return "Prediction could not be completed. Please check the server logs.", 500


# ==========================================
# Weather Code Description
# ==========================================

def get_weather_description(code):
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    try:
        return descriptions.get(int(code), "Weather condition unavailable")
    except (TypeError, ValueError):
        return "Weather condition unavailable"


# ==========================================
# Live Weather Dashboard
# ==========================================

@app.route("/weather", methods=["GET"])
def weather():
    city = request.args.get("city", "").strip()

    error = None
    current = None
    daily_rows = []
    place = None
    updated_at = None

    if city:
        try:
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                },
                timeout=10
            )
            geo_response.raise_for_status()

            locations = geo_response.json().get("results", [])

            if not locations:
                error = "Location not found. Try another city name."
            else:
                location = locations[0]
                latitude = location["latitude"]
                longitude = location["longitude"]

                place_parts = [
                    location.get("name"),
                    location.get("admin1"),
                    location.get("country")
                ]

                place = ", ".join(
                    dict.fromkeys(part for part in place_parts if part)
                )

                weather_response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": (
                            "temperature_2m,relative_humidity_2m,"
                            "apparent_temperature,precipitation,rain,"
                            "weather_code,wind_speed_10m"
                        ),
                        "daily": (
                            "weather_code,temperature_2m_max,"
                            "temperature_2m_min,"
                            "precipitation_probability_max,"
                            "precipitation_sum"
                        ),
                        "timezone": "auto",
                        "forecast_days": 5
                    },
                    timeout=10
                )
                weather_response.raise_for_status()

                weather_data = weather_response.json()
                raw_current = weather_data.get("current", {})

                current = {
                    "temperature": raw_current.get("temperature_2m"),
                    "feels_like": raw_current.get("apparent_temperature"),
                    "humidity": raw_current.get("relative_humidity_2m"),
                    "precipitation": raw_current.get("precipitation"),
                    "rain": raw_current.get("rain"),
                    "wind_speed": raw_current.get("wind_speed_10m"),
                    "condition": get_weather_description(
                        raw_current.get("weather_code")
                    )
                }

                updated_at = raw_current.get("time")

                raw_daily = weather_data.get("daily", {})

                dates = raw_daily.get("time", [])
                codes = raw_daily.get("weather_code", [])
                max_temps = raw_daily.get("temperature_2m_max", [])
                min_temps = raw_daily.get("temperature_2m_min", [])
                rain_probs = raw_daily.get(
                    "precipitation_probability_max", []
                )
                rain_totals = raw_daily.get("precipitation_sum", [])

                for i, date in enumerate(dates):
                    def get_daily_value(values):
                        return values[i] if i < len(values) else None

                    daily_rows.append({
                        "date": date,
                        "condition": get_weather_description(
                            get_daily_value(codes)
                        ),
                        "max_temp": get_daily_value(max_temps),
                        "min_temp": get_daily_value(min_temps),
                        "rain_probability": get_daily_value(rain_probs),
                        "precipitation": get_daily_value(rain_totals)
                    })

        except requests.RequestException:
            error = (
                "Unable to retrieve weather data. "
                "Check your internet connection and try again."
            )
        except (KeyError, IndexError, TypeError, ValueError):
            error = (
                "The weather service returned incomplete data. "
                "Please try again."
            )

    return render_template(
        "weather.html",
        city=city,
        place=place,
        current=current,
        daily_rows=daily_rows,
        updated_at=updated_at,
        error=error
    )


# ==========================================
# Automatic Monitoring Helpers
# ==========================================

def get_monitor_weather(latitude, longitude):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,rain,"
                "weather_code,wind_speed_10m"
            ),
            "hourly": "precipitation,precipitation_probability",
            "daily": (
                "weather_code,temperature_2m_max,"
                "temperature_2m_min,precipitation_probability_max,"
                "precipitation_sum"
            ),
            "forecast_days": 5,
            "timezone": "auto"
        },
        timeout=15
    )
    response.raise_for_status()
    return response.json()


def calculate_weather_indicator(weather_data):
    """Rainfall screening heuristic, not the XGBoost model."""
    current_data = weather_data.get("current", {})
    hourly = weather_data.get("hourly", {})

    times = hourly.get("time", [])
    rainfall = hourly.get("precipitation", [])
    probabilities = hourly.get("precipitation_probability", [])

    current_time = current_data.get("time")
    start = 0

    if current_time and times:
        for i, value in enumerate(times):
            if value >= current_time:
                start = i
                break

    next_rain = [
        float(value)
        for value in rainfall[start:start + 24]
        if value is not None
    ]

    next_probabilities = [
        float(value)
        for value in probabilities[start:start + 24]
        if value is not None
    ]

    rain_24h = round(sum(next_rain), 1)
    max_probability = (
        int(max(next_probabilities)) if next_probabilities else 0
    )

    if rain_24h >= 50 or (
        rain_24h >= 25 and max_probability >= 80
    ):
        category = "high"
        level = "High weather concern"
        explanation = (
            "Forecast rainfall is substantial. Check official local "
            "warnings and river or drainage conditions."
        )
    elif rain_24h >= 20 or (
        rain_24h >= 10 and max_probability >= 60
    ):
        category = "moderate"
        level = "Moderate weather concern"
        explanation = (
            "Forecast rainfall may warrant attention. Monitor official "
            "forecasts and local conditions."
        )
    else:
        category = "low"
        level = "Lower weather concern"
        explanation = (
            "Forecast rainfall is below this prototype's screening "
            "thresholds. Flooding may still occur for other reasons."
        )

    return {
        "level": level,
        "category": category,
        "rain_24h_mm": rain_24h,
        "max_rain_probability": max_probability,
        "explanation": explanation
    }


# ==========================================
# Automatic Monitoring Page
# ==========================================

@app.route("/monitor")
def automatic_monitor():
    return render_template("monitor.html")


# ==========================================
# Automatic Monitoring API
# ==========================================

@app.route("/api/monitor")
def automatic_monitor_data():
    try:
        latitude = float(request.args.get("lat", ""))
        longitude = float(request.args.get("lon", ""))
        location_name = request.args.get(
            "label", "Selected location"
        )[:120]

        if not (
            np.isfinite(latitude)
            and np.isfinite(longitude)
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            return jsonify({
                "error": "Invalid latitude or longitude."
            }), 400

        weather_data = get_monitor_weather(latitude, longitude)
        current = weather_data.get("current", {})
        daily_data = weather_data.get("daily", {})
        assessment = calculate_weather_indicator(weather_data)

        daily = []

        for i, date in enumerate(daily_data.get("time", [])):
            def daily_value(key):
                values = daily_data.get(key, [])
                return values[i] if i < len(values) else None

            daily.append({
                "date": date,
                "condition": get_weather_description(
                    daily_value("weather_code")
                ),
                "min_temp": daily_value("temperature_2m_min"),
                "max_temp": daily_value("temperature_2m_max"),
                "rain_probability": daily_value(
                    "precipitation_probability_max"
                ),
                "precipitation": daily_value("precipitation_sum")
            })

        result = {
            "location": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "updated_at": current.get("time"),
            "current": {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "rain": current.get("rain"),
                "wind_speed": current.get("wind_speed_10m"),
                "condition": get_weather_description(
                    current.get("weather_code")
                )
            },
            "assessment": assessment,
            "daily": daily,
            "disclaimer": (
                "This is a rainfall-based prototype indicator, not an "
                "ML-derived flood probability or official warning. It "
                "does not account for river levels, terrain, drainage, "
                "or local flood observations."
            )
        }

        # Keep automatic monitoring history separate from ML predictions.
        history_data = pd.read_csv(MONITOR_HISTORY_FILE)

        history_data.loc[len(history_data)] = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            location_name,
            latitude,
            longitude,
            assessment["rain_24h_mm"],
            assessment["max_rain_probability"],
            assessment["level"]
        ]

        history_data.to_csv(MONITOR_HISTORY_FILE, index=False)

        return jsonify(result)

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please provide valid coordinates."
        }), 400
    except requests.RequestException:
        return jsonify({
            "error": "Weather service unavailable. Check your internet connection."
        }), 502
    except Exception:
        app.logger.exception("Automatic monitoring failed")
        return jsonify({
            "error": "Monitoring could not be completed."
        }), 500


# ==========================================
# Prediction History
# ==========================================

@app.route("/history")
def history():
    history_data = pd.read_csv(HISTORY_FILE)

    table_html = history_data.to_html(
        classes="table table-striped table-bordered table-hover",
        index=False,
        justify="center",
        escape=True
    )

    return render_template(
        "history.html",
        tables=[table_html],
        titles=history_data.columns.values
    )


# ==========================================
# Automatic Monitoring History
# ==========================================

@app.route("/monitor-history")
def monitor_history():
    history_data = pd.read_csv(MONITOR_HISTORY_FILE)

    table_html = history_data.to_html(
        classes="table table-striped table-bordered table-hover",
        index=False,
        justify="center",
        escape=True
    )

    return render_template(
        "monitor_history.html",
        tables=[table_html],
        titles=history_data.columns.values
    )


# ==========================================
# Clear Prediction History
# ==========================================

@app.route("/clear_history", methods=["POST"])
def clear_history():
    pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(
        HISTORY_FILE,
        index=False
    )

    flash("Prediction history cleared successfully.", "success")
    return redirect(url_for("history"))


# ==========================================
# 404 Page
# ==========================================

@app.errorhandler(404)
def page_not_found(error):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Page Not Found | FloodSense AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container text-center py-5">
            <h1 class="display-1">404</h1>
            <h2>Page Not Found</h2>
            <p class="text-muted">The page you requested does not exist.</p>
            <a href="/" class="btn btn-primary">Back to Home</a>
        </div>
    </body>
    </html>
    """, 404


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
