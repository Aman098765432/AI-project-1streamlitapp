import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

# App title
st.title("📊 Customer Churn Prediction System")

st.write(
    "This application predicts whether a telecom customer "
    "is likely to leave the company using Machine Learning."
)

# Load model
if not os.path.exists("churn_model.joblib"):
    st.error(
        "Model file not found! Please upload "
        "'churn_model.joblib' to your GitHub repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Churn Prediction", "Model Information"]
)

# HOME PAGE
if page == "Home":

    st.header("Welcome to Customer Churn Prediction")

    st.write("""
    This Machine Learning application helps telecom companies
    identify customers who may leave their services.

    The project uses:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Machine Learning Model Evaluation
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Project", "Customer Churn")
    col2.metric("ML Models", "3")
    col3.metric("Prediction", "Yes / No")

# PREDICTION PAGE
elif page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    st.write("Enter customer information below:")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    # Prediction button
    if st.button("Predict Customer Churn"):

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Apply the same encoding as training
        customer_encoded = pd.get_dummies(
            customer,
            drop_first=False
        )

        # Align with model's training features
        if not hasattr(model, "feature_names_in_"):
            st.error(
                "The saved model does not contain feature names. "
                "Please save the model again using the "
                "DataFrame-based training code."
            )
            st.stop()

        model_features = model.feature_names_in_

        # Rebuild dummy columns using training categories.
        # With one customer, drop_first=True would remove
        # some required categories, so explicitly encode each
        # value into its trained feature column.
        aligned = pd.DataFrame(
            0.0,
            index=customer.index,
            columns=model_features
        )

        numeric_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for column in numeric_columns:
            if column in aligned.columns:
                aligned[column] = customer[column].astype(float)

        categorical_columns = [
            c for c in customer.columns
            if c not in numeric_columns
        ]

        for column in categorical_columns:
            value = str(customer.at[0, column])
            dummy_name = f"{column}_{value}"

            if dummy_name in aligned.columns:
                aligned[dummy_name] = 1.0

        try:
            prediction = model.predict(aligned)[0]

            probability = model.predict_proba(aligned)[0][1]

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:
                    st.error("⚠️ Customer May Leave")
                else:
                    st.success("✅ Customer Likely to Stay")

            with col2:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(float(probability))

            if probability >= 0.7:
                st.warning("High Churn Risk")
            elif probability >= 0.4:
                st.info("Medium Churn Risk")
            else:
                st.success("Low Churn Risk")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# MODEL INFORMATION PAGE
elif page == "Model Information":

    st.header("Machine Learning Model Information")

    st.write("""
    ### Models Used

    1. Logistic Regression
    2. Random Forest
    3. XGBoost

    ### Dataset

    IBM Telco Customer Churn Dataset

    ### Machine Learning Steps

    - Data Cleaning
    - Feature Engineering
    - Train-Test Split
    - Cross Validation
    - Hyperparameter Tuning
    - Model Comparison
    - Model Deployment
    """)

    st.success("Machine Learning Model Loaded Successfully!")

# Footer
st.divider()
st.caption("Customer Churn Prediction | Machine Learning Project")
