
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

# --------------------------------------------------
# 1. LOAD MODEL AND METADATA
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "model.joblib"

METADATA_PATH = APP_DIR / "model_metadata.json"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:

        metadata = json.load(file)

    return model, metadata


model, metadata = load_model()


# --------------------------------------------------
# 2. STREAMLIT APPLICATION
# --------------------------------------------------


st.title("Tourism Package Purchase Prediction")

st.write(
    """
    This application predicts whether a customer is
    likely to purchase the Wellness Tourism Package.

    Enter the customer details below to generate
    a prediction.
    """
)

st.divider()


# --------------------------------------------------
# 3. COLLECT CUSTOMER INPUTS
# --------------------------------------------------

input_values = {}

feature_metadata = metadata["feature_metadata"]

st.subheader("Customer Information")

for feature, details in feature_metadata.items():

    # CityTier: Integer dropdown (1, 2, 3)
    if feature == "CityTier":

        input_values[feature] = st.selectbox(
            feature,
            options=[1, 2, 3]
        )

    # Passport: Yes / No dropdown
    elif feature == "Passport":

        value = st.selectbox(
            feature,
            options=["Yes", "No"]
        )

        input_values[feature] = 1 if value == "Yes" else 0

    # OwnCar: Yes / No dropdown
    elif feature == "OwnCar":

        value = st.selectbox(
            feature,
            options=["Yes", "No"]
        )

        input_values[feature] = 1 if value == "Yes" else 0

    # PitchSatisfactionScore: Integer between 1 and 5
    elif feature == "PitchSatisfactionScore":

        input_values[feature] = st.number_input(
            feature,
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            format="%d"
        )

    # All other numerical features: Integer
    elif details["type"] == "numerical":

        input_values[feature] = st.number_input(
            feature,
            value=int(details["default"]),
            step=1,
            format="%d"
        )

    # Remaining categorical features: Dropdown
    else:

        options = details["options"]

        if options:

            input_values[feature] = st.selectbox(
                feature,
                options=options
            )

        else:

            input_values[feature] = st.text_input(
                feature
            )


# --------------------------------------------------
# 4. CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame(
    [input_values]
)


# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Purchase",
    type="primary"
):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.success(
            "The customer is predicted to purchase "
            "the Wellness Tourism Package."
        )

    else:

        st.info(
            "The customer is predicted not to purchase "
            "the Wellness Tourism Package."
        )

    st.write(
        "Predicted Class:",
        int(prediction)
    )

    st.write(
        "Purchase Probability:",
        round(float(probability), 4)
    )
