import os
import pickle
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_FILE = "heart_disease_final_model.pkl"

FINAL_THRESHOLD = 0.5183500182756475

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope"
]


# ============================================================
# LOAD FINAL MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the final trained model and threshold from the pickle file.
    """

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"{MODEL_FILE} was not found in the project folder."
        )

    with open(MODEL_FILE, "rb") as file:
        package = pickle.load(file)

    return package["model"], package["threshold"]


try:
    model, threshold = load_model()

except Exception as error:
    st.error("Unable to load the final ML model.")
    st.exception(error)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_binary_file_downloader_html(dataframe):
    """
    Create a downloadable CSV link for a DataFrame.
    """

    csv_data = dataframe.to_csv(index=False)

    encoded_data = base64.b64encode(
        csv_data.encode()
    ).decode()

    href = (
        f'<a href="data:file/csv;base64,{encoded_data}" '
        f'download="heart_disease_predictions.csv">'
        f'Download Predictions CSV File'
        f'</a>'
    )

    return href


def clean_input_data(dataframe):
    """
    Apply the same invalid-zero handling used during model training.

    Cholesterol = 0 and RestingBP = 0 are treated as missing values.
    The model's internal imputer then handles them.
    """

    data = dataframe.copy()

    if "Cholesterol" in data.columns:
        data.loc[
            data["Cholesterol"] == 0,
            "Cholesterol"
        ] = np.nan

    if "RestingBP" in data.columns:
        data.loc[
            data["RestingBP"] == 0,
            "RestingBP"
        ] = np.nan

    return data


def predict_heart_disease(dataframe):
    """
    Generate prediction and model score using the final model.
    """

    cleaned_data = clean_input_data(dataframe)

    model_score = model.predict_proba(
        cleaned_data
    )[:, 1]

    predictions = (
        model_score >= threshold
    ).astype(int)

    return predictions, model_score


def prediction_label(prediction):
    """
    Convert numerical prediction into human-readable text.
    """

    if prediction == 1:
        return "Heart Disease Detected"

    return "No Heart Disease Detected"


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("❤️ Heart Disease Predictor")

st.write(
    "Machine-learning based heart disease prediction "
    "using a tuned Random Forest model."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Predict",
        "Bulk Predict",
        "Model Information"
    ]
)


# ============================================================
# TAB 1 — SINGLE PATIENT PREDICTION
# ============================================================

with tab1:

    st.header("Single Patient Prediction")

    st.info(
        "Enter the patient's clinical information below "
        "to generate a prediction."
    )

    # --------------------------------------------------------
    # INPUT FIELDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=150,
            value=50,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-Anginal Pain",
                "Asymptomatic"
            ]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=0,
            max_value=300,
            value=120,
            step=1
        )

        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=0,
            max_value=1000,
            value=200,
            step=1
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [
                "<= 120 mg/dl",
                "> 120 mg/dl"
            ]
        )

    with col2:

        resting_ecg = st.selectbox(
            "Resting ECG Results",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=202,
            value=150,
            step=1
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            [
                "Yes",
                "No"
            ]
        )

        oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ]
        )


    # --------------------------------------------------------
    # CONVERT UI LABELS TO ORIGINAL DATASET VALUES
    # --------------------------------------------------------

    sex_value = {
        "Male": "M",
        "Female": "F"
    }[sex]

    chest_pain_value = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP",
        "Asymptomatic": "ASY"
    }[chest_pain]

    fasting_bs_value = {
        "<= 120 mg/dl": 0,
        "> 120 mg/dl": 1
    }[fasting_bs]

    resting_ecg_value = {
        "Normal": "Normal",
        "ST-T Wave Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH"
    }[resting_ecg]

    exercise_angina_value = {
        "Yes": "Y",
        "No": "N"
    }[exercise_angina]

    st_slope_value = {
        "Upsloping": "Up",
        "Flat": "Flat",
        "Downsloping": "Down"
    }[st_slope]


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_value],
        "ChestPainType": [chest_pain_value],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs_value],
        "RestingECG": [resting_ecg_value],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_value],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_value]
    })


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Heart Disease",
        type="primary"
    ):

        predictions, scores = predict_heart_disease(
            input_data
        )

        prediction = predictions[0]
        score = scores[0]

        result = prediction_label(prediction)


        st.divider()

        st.subheader("Prediction Result")


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                f"⚠️ {result}"
            )

        else:

            st.success(
                f"✅ {result}"
            )


        # ----------------------------------------------------
        # MODEL SCORE
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Model Score",
                f"{score * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.2f}%"
            )


        st.caption(
            "The model score is the Random Forest's estimated "
            "score for the heart-disease class. The final "
            "classification is based on the optimized threshold."
        )


# ============================================================
# TAB 2 — BULK PREDICTION
# ============================================================

with tab2:

    st.header("Bulk Prediction")

    st.write(
        "Upload a CSV file containing patient records "
        "to generate predictions for multiple patients."
    )


    st.info(
        """
        ### Required Features

        Your CSV must contain these 11 columns:

        `Age, Sex, ChestPainType, RestingBP, Cholesterol,
        FastingBS, RestingECG, MaxHR, ExerciseAngina,
        Oldpeak, ST_Slope`

        ### Supported categorical formats

        The application accepts the original dataset values:

        - Sex → `M` / `F`
        - ChestPainType → `ATA` / `NAP` / `ASY` / `TA`
        - RestingECG → `Normal` / `ST` / `LVH`
        - ExerciseAngina → `Y` / `N`
        - ST_Slope → `Up` / `Flat` / `Down`

        It also accepts the numeric conventions used by the
        original application.
        """
    )


    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            bulk_data = pd.read_csv(
                uploaded_file
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                bulk_data.head(),
                use_container_width=True
            )


            # ------------------------------------------------
            # CHECK REQUIRED COLUMNS
            # ------------------------------------------------

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in bulk_data.columns
            ]


            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                st.stop()


            # ------------------------------------------------
            # PREPARE MODEL DATA
            # ------------------------------------------------

            prediction_data = bulk_data[
                FEATURE_COLUMNS
            ].copy()


            # ------------------------------------------------
            # SUPPORT OLD NUMERIC ENCODINGS
            # ------------------------------------------------

            # Sex
            prediction_data["Sex"] = (
                prediction_data["Sex"]
                .replace({
                    0: "M",
                    1: "F",
                    "Male": "M",
                    "Female": "F"
                })
            )


            # Chest Pain Type
            prediction_data["ChestPainType"] = (
                prediction_data["ChestPainType"]
                .replace({
                    0: "ATA",
                    1: "NAP",
                    2: "ASY",
                    3: "TA",
                    "Typical Angina": "TA",
                    "Atypical Angina": "ATA",
                    "Non-Anginal Pain": "NAP",
                    "Asymptomatic": "ASY"
                })
            )


            # Fasting Blood Sugar
            prediction_data["FastingBS"] = (
                prediction_data["FastingBS"]
                .replace({
                    "> 120 mg/dl": 1,
                    "<= 120 mg/dl": 0
                })
            )

            prediction_data["FastingBS"] = pd.to_numeric(
                prediction_data["FastingBS"],
                errors="coerce"
            )


            # Resting ECG
            prediction_data["RestingECG"] = (
                prediction_data["RestingECG"]
                .replace({
                    0: "Normal",
                    1: "ST",
                    2: "LVH",
                    "ST-T Wave Abnormality": "ST",
                    "Left Ventricular Hypertrophy": "LVH"
                })
            )


            # Exercise Angina
            prediction_data["ExerciseAngina"] = (
                prediction_data["ExerciseAngina"]
                .replace({
                    0: "N",
                    1: "Y",
                    "Yes": "Y",
                    "No": "N"
                })
            )


            # ST Slope
            prediction_data["ST_Slope"] = (
                prediction_data["ST_Slope"]
                .replace({
                    0: "Up",
                    1: "Flat",
                    2: "Down",
                    "Upsloping": "Up",
                    "Downsloping": "Down"
                })
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            predictions, scores = predict_heart_disease(
                prediction_data
            )


            # ------------------------------------------------
            # CREATE OUTPUT
            # ------------------------------------------------

            output_data = bulk_data.copy()

            output_data["Prediction"] = predictions

            output_data["Prediction Result"] = [
                prediction_label(prediction)
                for prediction in predictions
            ]

            output_data["Model Score (%)"] = (
                scores * 100
            ).round(2)


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            st.success(
                f"Predictions generated for "
                f"{len(output_data)} patients."
            )

            st.subheader("Prediction Results")

            st.dataframe(
                output_data,
                use_container_width=True
            )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.markdown(
                get_binary_file_downloader_html(
                    output_data
                ),
                unsafe_allow_html=True
            )


        except Exception as error:

            st.error(
                "Unable to process the uploaded CSV file."
            )

            st.exception(error)


    else:

        st.info(
            "Upload a CSV file to generate predictions."
        )


# ============================================================
# TAB 3 — MODEL INFORMATION
# ============================================================

with tab3:

    st.header("Model Information")


    st.subheader("Final Model")

    st.write(
        "**Tuned Random Forest Classifier**"
    )

    st.write(
        "The final model was selected after comparing "
        "Logistic Regression, Decision Tree, Random Forest "
        "and SVM models, followed by hyperparameter tuning."
    )


    # --------------------------------------------------------
    # FINAL MODEL METRICS
    # --------------------------------------------------------

    st.subheader("Final Model Performance")


    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.metric(
            "Accuracy",
            "89.13%"
        )

        st.metric(
            "Recall",
            "93.14%"
        )


    with metric_col2:

        st.metric(
            "Precision",
            "87.96%"
        )

        st.metric(
            "F1 Score",
            "90.48%"
        )


    with metric_col3:

        st.metric(
            "ROC-AUC",
            "93.01%"
        )

        st.metric(
            "Specificity",
            "84.15%"
        )


    st.write(
        f"**Optimized Decision Threshold:** "
        f"{threshold:.4f}"
    )


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")


    comparison_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Tuned Logistic Regression",
            "Tuned Decision Tree",
            "Tuned Random Forest",
            "Tuned SVM"
        ],

        "Accuracy": [
            0.8859,
            0.7772,
            0.8804,
            0.8696,
            0.8913,
            0.8207,
            0.8859,
            0.8641
        ],

        "Recall": [
            0.9118,
            0.7647,
            0.9020,
            0.9020,
            0.9118,
            0.7941,
            0.9314,
            0.8333
        ],

        "F1 Score": [
            0.8986,
            0.7919,
            0.8932,
            0.8846,
            0.9029,
            0.8308,
            0.9005,
            0.8718
        ],

        "ROC-AUC": [
            0.9329,
            0.7787,
            0.9328,
            0.9411,
            0.9346,
            0.8890,
            0.9301,
            0.9268
        ]
    })


    comparison_display = comparison_data.copy()

    for column in [
        "Accuracy",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        comparison_display[column] = (
            comparison_display[column] * 100
        ).round(2)


    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    fig_accuracy = px.bar(
        comparison_display,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        labels={
            "Accuracy": "Accuracy (%)"
        }
    )

    fig_accuracy.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig_accuracy,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RECALL CHART
    # --------------------------------------------------------

    fig_recall = px.bar(
        comparison_display,
        x="Model",
        y="Recall",
        title="Model Recall Comparison",
        labels={
            "Recall": "Recall (%)"
        }
    )

    fig_recall.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig_recall,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MODEL PIPELINE
    # --------------------------------------------------------

    st.subheader("ML Pipeline")

    st.code(
        """
Raw Patient Data
       ↓
Invalid Zero Handling
       ↓
Numerical Imputation + Scaling
       +
Categorical Imputation + One-Hot Encoding
       ↓
Tuned Random Forest
       ↓
Optimized Threshold = 0.51835
       ↓
Heart Disease Prediction
        """,
        language="text"
    )


    st.warning(
        "This application is intended for educational and "
        "machine-learning demonstration purposes. It is not "
        "a clinically validated diagnostic system."
    )