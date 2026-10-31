
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

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

    if details["type"] == "numerical":

        input_values[feature] = st.number_input(
            feature,
            value=float(details["default"])
        )

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
