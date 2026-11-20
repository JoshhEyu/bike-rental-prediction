
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_bike_rental_model.pkl"
)

st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="wide"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    feature_columns = saved_model["features"]

    return model, feature_columns


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🚲 Bike Rental Demand Predictor")

st.write(
    """
    Predict hourly bicycle rental demand using a trained
    Gradient Boosting machine learning model.
    
    Enter the date, time, and environmental conditions
    to estimate the expected number of bicycle rentals.
    """
)

st.divider()


# ==========================================
# 4. LOAD MODEL
# ==========================================

try:

    model, feature_columns = load_model()

except Exception as error:

    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ==========================================
# 5. USER INPUTS
# ==========================================

st.subheader("Prediction Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # --------------------------------------
    # TEMPORAL INFORMATION
    # --------------------------------------

    with col1:

        st.markdown("### 📅 Date and Time")

        year = st.selectbox(
            "Year",
            [2011, 2012],
            index=1
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=5
        )

        day = st.slider(
            "Day of Month",
            1,
            31,
            15
        )

        hour = st.slider(
            "Hour of Day",
            0,
            23,
            17
        )

        weekday = st.selectbox(
            "Day of Week",
            [0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday"
            ][x]
        )

        season = st.selectbox(
            "Season",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Spring",
                2: "Summer",
                3: "Fall",
                4: "Winter"
            }[x]
        )

    # --------------------------------------
    # WEATHER INFORMATION
    # --------------------------------------

    with col2:

        st.markdown("### 🌤️ Weather Conditions")

        weather = st.selectbox(
            "Weather Situation",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear or partly cloudy",
                2: "Mist or cloudy",
                3: "Light rain or snow",
                4: "Heavy rain or snow"
            }[x]
        )

        temp = st.slider(
            "Temperature (normalized)",
            0.0, 1.0, 0.5, 0.01
        )

        atemp = st.slider(
            "Feeling Temperature (normalized)",
            0.0, 1.0, 0.5, 0.01
        )

        humidity = st.slider(
            "Humidity (normalized)",
            0.0, 1.0, 0.5, 0.01
        )

        windspeed = st.slider(
            "Wind Speed (normalized)",
            0.0, 1.0, 0.2, 0.01
        )

        holiday = st.selectbox(
            "Is it a holiday?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        workingday = st.selectbox(
            "Is it a working day?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.info(
        "Temperature, feeling temperature, humidity, "
        "and wind speed use normalized values between 0 and 1."
    )

    submitted = st.form_submit_button(
        "Predict Rental Demand",
        use_container_width=True
    )


# ==========================================
# 6. PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame([{

        "yr": year - 2011,
        "holiday": holiday,
        "workingday": workingday,
        "temp": temp,
        "atemp": atemp,
        "hum": humidity,
        "windspeed": windspeed,
        "day": day,
        "season": season,
        "mnth": month,
        "hr": hour,
        "weekday": weekday,
        "weathersit": weather

    }])

    # Apply the same one-hot encoding used during training

    categorical_columns = [
        "season",
        "mnth",
        "hr",
        "weekday",
        "weathersit"
    ]

    encoded_input = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True
    )

    # Match the exact feature structure used by the model

    encoded_input = encoded_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    try:

        prediction = model.predict(encoded_input)[0]

        prediction = max(0, round(float(prediction)))

        st.divider()

        st.subheader("Prediction Results")

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Hourly Bicycle Rentals",
            value=f"{prediction:,} bicycles"
        )

        st.caption(
            "This is an estimated rental count based on "
            "the information provided."
        )

        with st.expander("View Submitted Information"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# ==========================================
# 7. MODEL INFORMATION
# ==========================================

st.divider()

st.subheader("About the Machine Learning Model")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "Gradient Boosting")
col2.metric("Test R² Score", "78.64%")
col3.metric("Test MAE", "71.60")

st.write(
    """
    The application uses the final tuned Gradient Boosting
    regression model developed using the UCI Bike Sharing
    Dataset.
    """
)

st.caption(
    "Machine Learning Assignment – Project 2 | November 2026"
)
