import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64       #for encoding-decoding the file that we are taking as input from website

#Function to create a downloadable link for a Dataframe as a CSV file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'
    return href


st.title("Heart Disease Predictor")        #Title of the web app
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information']) #Creating three tabs for different functionalities

with tab1:      #First tab for single prediction
    age = st.number_input("Age (years)", min_value=1, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0, max_value=1000)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"],)
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left ventricular hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Converting categorical inputs to numerical values
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left ventricular hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Creating a DataFrame for the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],         
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

#Frontend below line
algonames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
modelnames = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'gridrf.pkl']

#define a function to load the model from the pickle file

predictions = []
def predict_heart_disease(data):
    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions  
      
#Create a submit button to make the prediction

   
if st.button("Submit"):         
        st.subheader('Results...')
        st.markdown('---------------------------------------')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No Heart Disease Detected")
            else:
                st.write("Heart Disease Detected")
            st.markdown('---------------------------------------')



with tab2:      #Second tab for bulk prediction using CSV file
    st.title("Upload CSV File")

    st.subheader("Instructions to note before uploading the file:")
    st.info("""
        1. No NaN values allowed
        2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'). \n
        3. Check the spelling of feature names.
        4. Feature values conventions: \n
            -Age: Integer values [years] \n
            -Sex: sex of the patient (0 = male, 1 = female) \n
            -ChestPainType: chest pain type (3 = Typical Angina, 0 = Atypical Angina, 1 = Non-Anginal Pain, 2 = Asymptomatic) \n 
            -RestingBP: resting blood pressure [mm Hg] \n
            -Cholesterol: serum cholesterol [mm/dl] \n  
            -FastingBS: fasting blood sugar (1 = if FastingBS > 120 mg/dl, 0 = otherwise) \n
            -RestingECG: resting electrocardiographic results (0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left ventricular hypertrophy) \n
            -MaxHR: maximum heart rate achieved [numeric values between 60 and 202] \n
            -ExerciseAngina: exercise-induced angina (1 = yes; 0 = no) \n
            -Oldpeak: oldpeak = ST depression induced by exercise relative to rest [numeric values between 0.0 and 10.0] \n
            -ST_Slope: the slope of the peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping) \n


""")
    
    # Create a file uploader in the sidebar
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
         # Read the CSV file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    model = pickle.load(open('LogisticRegression.pkl','rb')) #Loading a default model for bulk prediction

    #Ensure that the input Dataframe matches the expected columns and format
    expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
     
    if set(expected_columns).issubset(input_data.columns):
        input_data['Prediction LR'] = ''

        for i in range(len(input_data)):
            arr = input_data.iloc[i,:-1].values
            input_data['Prediction LR'][i] = model.predict([arr])[0]
        input_data.to_csv('PredictedHeartLR.csv')

        # Display the DataFrame with predictions
        st.subheader("Predictions:")
        st.write(input_data)

        #Create a button to download the predictions as a CSV file
        st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
    else:
        st.warning("Please  make sure the uploaded CSV file has the correct columns. ")

else:
    st.info("Upload a CSV file to get predictions.")


with tab3:      #Third tab for model information
    import plotly.express as px
    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22, 'Grid Random Forest': 89.75}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)
