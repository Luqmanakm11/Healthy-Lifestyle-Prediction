import streamlit as st
import joblib
import pandas as pd
import dill

# Load the preprocessing function
try:
    with open("D:/FYP/load_and_preprocess_real_data.pkl", "rb") as f:
        load_and_preprocess_real_data = dill.load(f)
    st.success("Preprocessing function loaded successfully.")
except Exception as e:
    st.error(f"Error loading preprocessing function: {str(e)}")

# Load the trained model
try:
    model = joblib.load("D:/FYP/healthy_lifestyle_model.pkl")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Streamlit App
st.title("Healthy Lifestyle Prediction")
st.write("Fill in the details below to predict your healthy lifestyle score and status.")

# Input fields
mood = st.text_input("Mood", placeholder="e.g., happy")
workout_type = st.text_input("Workout Type", placeholder="e.g., cardio")
weather_conditions = st.text_input("Weather Conditions", placeholder="e.g., sunny")
location = st.text_input("Location", placeholder="e.g., indoor")
steps = st.number_input("Steps", min_value=0.0, step=1.0)
calories_burned = st.number_input("Calories Burned", min_value=0.0, step=0.1)
distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.1)
active_minutes = st.number_input("Active Minutes", min_value=0.0, step=1.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1)
heart_rate_avg = st.number_input("Heart Rate (Average)", min_value=0.0, step=1.0)

# Prediction
if st.button("Predict"):
    try:
        # Prepare the data
        data = {
            'mood': mood,
            'workout_type': workout_type,
            'weather_conditions': weather_conditions,
            'location': location,
            'steps': steps,
            'calories_burned': calories_burned,
            'distance_km': distance_km,
            'active_minutes': active_minutes,
            'sleep_hours': sleep_hours,
            'heart_rate_avg': heart_rate_avg
        }

        # Convert input data into a DataFrame
        df = pd.DataFrame([data])
        st.write("Input Data:", df)  # Debugging: Show input data to the user

        # Preprocess the input data
        preprocessed_data, _ = load_and_preprocess_real_data(df)
        st.write("Preprocessed Data:", preprocessed_data)  # Debugging: Show preprocessed data

        # Predict using the loaded model
        prediction = model.predict(preprocessed_data)

        # Check prediction format and decode to 'healthy' or 'unhealthy'
        if isinstance(prediction[0], (float, int)):  # If numeric score
            health_status = 'healthy' if prediction[0] > 0.5 else 'unhealthy'
            st.success(f"Healthy Score: {prediction[0]:.2f}")
        elif isinstance(prediction[0], str):  # If categorical prediction
            health_status = prediction[0]
        else:
            raise ValueError("Unexpected prediction output format.")

        st.success(f"Health Status: {health_status.capitalize()}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug Info:", e)
