import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline committed to tourism_project/deployment/
MODEL_PATH = "tourism_project/deployment/best_tourism_model_v1.joblib"
model = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter customer and sales interaction details below to predict whether the customer will purchase the Wellness Tourism Package.")

# --- Customer Details ---
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=22500.0, step=500.0)

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
    number_of_children_visiting = st.number_input("Number of Children Visiting (Below Age 5)", min_value=0, max_value=5, value=1, step=1)
    number_of_trips = st.number_input("Number of Trips Annually", min_value=1, max_value=30, value=3, step=1)
    passport_input = st.selectbox("Valid Passport", ["No", "Yes"])
    own_car_input = st.selectbox("Owns a Car", ["No", "Yes"])

# --- Customer Interaction Data ---
st.subheader("Sales Interaction Details")

col3, col4 = st.columns(2)

with col3:
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col4:
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    number_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=4, step=1)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=150, value=15, step=1)

# Convert Yes/No selections to binary (0/1) matching training data
passport = 1 if passport_input == "Yes" else 0
own_car = 1 if own_car_input == "Yes" else 0

# Assemble input DataFrame with exact column names expected by ColumnTransformer
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

# Predict using the same 0.45 classification threshold from train.py
CLASSIFICATION_THRESHOLD = 0.45

if st.button("Predict Purchase Likelihood"):
    purchase_proba = model.predict_proba(input_data)[0, 1]
    prediction = 1 if purchase_proba >= CLASSIFICATION_THRESHOLD else 0

    st.subheader("Prediction Result")
    st.metric(label="Purchase Probability", value=f"{purchase_proba:.2%}")

    if prediction == 1:
        st.success("High Likelihood: The customer is **likely to purchase** the Wellness Tourism Package.")
    else:
        st.warning("Low Likelihood: The customer is **unlikely to purchase** the Wellness Tourism Package.")
