import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained classification model
with open('churn.pkl', 'rb') as file:
    model = pickle.load(file)

#Set up the Streamlit app
st.title('Customer Churn Predicton')

# Function to preprocess and scale the input features
def preprocess_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

# Function to make predictions
def predict(features):
    scaled_features = preprocess_features(features)
    prediction = model.predict(scaled_features)
    return prediction

# Create input fields for the 27 features
# feature_columns = ['feature' + str(i) for i in range(1, 28)]
feature_columns = [
     'Call Failure','Complains','Subscription  Length','Charge  Amount','Seconds of Use','Frequency of use','Frequency of SMS',	'Distinct Called Numbers','Age Group','Tariff Plan','Status','Customer Value'
]
feature_values = []
for column in feature_columns:
    value = st.number_input(column, step=0.01)
    feature_values.append(value)

# Make prediction when the user clicks the "Predict" button
if st.button('Predict'):
    features = pd.DataFrame([feature_values], columns=feature_columns)
    prediction = predict(features)
    
    # Convert the prediction to a readable label
    if prediction[0] == 1:
        result = 'Customer will leave the Service'
    else:
        result = 'Customer will continue the Service'
    
    # Display the prediction result
    st.write('Prediction:',result)
