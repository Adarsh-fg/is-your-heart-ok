import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Streamlit app
st.title('Heart Disease Prediction')
st.subheader('Enter the value of the report')

# input fields for the user

age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex',['M','F']) 
chest_pain = st.selectbox('Chest Pain Type',['ATA','NAP','ASY','TA'])
rest_bp = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
cholestrol = st.number_input('Cholesterol', min_value=0, max_value=700, value=200)
fbs = st.selectbox('Fasting Blood Sugar',['0','1'])
resting_ecg = st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_heart_rate = st.number_input('Max Heart Rate', min_value=50, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina',['N','Y'])
oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
slope = st.selectbox('Slope',['Up','Flat','Down'])

# convert categorical data to numerical data
input_data = {'Age':age,
              'Sex':sex,
              'ChestPainType':chest_pain,
              'RestingBP':rest_bp,
              'Cholestrol':cholestrol,
              'FastingBS':fbs,
              'RestingECG':resting_ecg,
              'MaxHR':max_heart_rate,
              'ExerciseAngina':exercise_angina,
              'Oldpeak':oldpeak,
              'ST_Slope':slope}

# Convert input data to DataFrame
new_data = pd.DataFrame([input_data]) 

# Load saved LabelEncoders
sex_encoder = LabelEncoder()
sex_encoder.classes_ = np.array(['F','M'])

chest_pain_encoder = LabelEncoder()
chest_pain_encoder.classes_ = np.array(['ATA','NAP','ASY','TA'])

resting_ecg_encoder = LabelEncoder()
resting_ecg_encoder.classes_ = np.array(['Normal','ST','LVH'])

exercise_angina_encoder = LabelEncoder()
exercise_angina_encoder.classes_ = np.array(['N','Y'])

st_slope_encoder = LabelEncoder()
st_slope_encoder.classes_ = np.array(['Up','Flat','Down'])

# Apply label encoding to categorical columns
new_data['Encoder_Sex'] = sex_encoder.transform(new_data['Sex'])
new_data['Encoder_ChestPainType'] = chest_pain_encoder.transform(new_data['ChestPainType'])
new_data['Encoder_RestingECG'] = resting_ecg_encoder.transform(new_data['RestingECG'])
new_data['Encoder_ExerciseAngina'] = exercise_angina_encoder.transform(new_data['ExerciseAngina'])
new_data["Encoder_ST_Slope"] = st_slope_encoder.transform(new_data["ST_Slope"])

# Drop original columns as they are already encoded
new_data.drop(["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], axis=1, inplace=True)

# Load the saved features list
df = pd.read_csv("features.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']

# Reindex to match the original column order
new_data = new_data.reindex(columns=columns_list, fill_value=0)

# Load the saved scaler
with open('Scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pkl.load(scaler_file)

# Scale the new data
scaled_data = pd.DataFrame(loaded_scaler.transform(new_data), columns=columns_list)

# Load the RandomForest model
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pkl .load(model_file)

# Make predictions
prediction = loaded_model.predict(scaled_data)

#  Make predictions only when the button is clicked
if st.button('Predict'):
    prediction = loaded_model.predict(scaled_data)

    # Output the prediction
    if prediction[0] == 1:
        st.error("Prediction: Heart Disease Present")
    else:
        st.success("Prediction: No Heart Disease")
