import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Load the saved model and the original dataset
model = joblib.load('model_final.joblib')
# Load the data
df_average_income = pd.read_csv('ori.csv')

# Define the inputs
location = st.text_input('Location (City)')
ownership = st.selectbox('Type of ownership', df_average_income['Type of ownership'].unique())
industry = st.selectbox('Industry', df_average_income['Industry'].unique())
sector = st.selectbox('Sector', df_average_income['Sector'].unique())
age = st.slider('Age', 18, 100, 25)
experience = st.slider('Years of Experience', 0, 50, 5)
job_title = st.selectbox('Categorized Job Title', df_average_income['Categorized Job Title'].unique())

# Make a prediction if all inputs are valid
if location and ownership and industry and sector and job_title:
    input_data = {
        'Location (City)': location,
        'Type of ownership': ownership,
        'Industry': industry,
        'Sector': sector,
        'Age': age,
        'Years of Experience': experience,
        'Categorized Job Title': job_title
    }

    input_df = pd.DataFrame(input_data, index=[0])
    df_encoded = pd.get_dummies(df_average_income, columns=['Location (City)', 'Type of ownership', 'Industry', 'Sector', 'Categorized Job Title'])
    input_df = input_df.reindex(columns=df_encoded.columns, fill_value=0)
    input_df = input_df[df_encoded.columns]
    input_df = input_df.drop('Average Take Home Income', axis=1)
    prediction = model.predict(input_df).item()

    # Display the prediction
    st.write('Predicted Take Home Income:', prediction)
else:
    st.write('Please enter all inputs')