import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model from the joblib file
model = joblib.load('model.joblib')

# Define the input form for the Streamlit app
st.write('Enter the following inputs:')
population = st.number_input('Population', min_value=0, max_value=1000000000, value=0, step=1000)
job_title = st.selectbox('Job Title', ['Data Scientist', 'Data Analyst', 'Data Engineer', 'Other'])

# Define the input features
input_features = pd.DataFrame({'Population': [population],
                               'Categorized Job Title_Data Analyst': [0],
                               'Categorized Job Title_Data Engineer': [0],
                               'Categorized Job Title_Data Scientist': [0],
                               'Categorized Job Title_Other': [0]})

# Set the appropriate job title feature to 1 based on user input
input_features['Categorized Job Title_' + job_title] = 1

# Make a prediction based on the input values
prediction = model.predict(input_features)

# Display the predicted salary
st.write('The predicted salary is: $', round(prediction[0],2))