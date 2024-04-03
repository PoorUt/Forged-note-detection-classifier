import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


with open('classifier.pkl', 'rb') as file:
    model = pickle.load(file)



# Create title

st.title("Bank Note Authentication Predictor")


# Subtitle
st.markdown("<h4>This App predicts the authentication of a bank note, by analyzing images of real and forged bank notes. A brief description of the data is provided below.</h4>", unsafe_allow_html=True)

# Description

st.markdown("""
    <div style='background-color: #ffff99; padding: 10px; border-radius: 5px;'>
        About Dataset ( From UCI Machine Learning Repository):

Data were extracted from images that were taken from genuine 
and forged banknote-like specimens. For digitization, an 
industrial camera usually used for print inspection was used.

Attribute Information:

1-Variance of Wavelet Transformed image (continuous)\n
2-Skewness of Wavelet Transformed image (continuous)\n
3-Curtosis of Wavelet Transformed image (continuous)\n
4-Entropy of image (continuous)\n
5-Class (integer)\n

Acknowledgements (From UCI Machine Learning Repository):

Owner of database: Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)
Donor of database: Helene DÃ¶rksen (University of Applied Sciences, Ostwestfalen-Lippe, helene.doerksen '@' hs-owl.de)

   
""", unsafe_allow_html=True)




# Input sliders for each feature
variance = st.slider('Variance', -10.0, 10.0, 0.0)
skewness = st.slider('Skewness', -15.0, 15.0, 0.0)
curtosis = st.slider('Curtosis', -20.0, 20.0, 0.0)
entropy = st.slider('Entropy', -10.0, 10.0, 0.0)



# Display the current values of the sliders
st.write('Current values:')
st.write('Variance:', variance)
st.write('Skewness:', skewness)
st.write('Curtosis:', curtosis)
st.write('Entropy:', entropy)



# Predict button with custom CSS
predict_button_html = """
    <style>
        .predict-button {
            font-size: 18px;
            color: white;
            background-color: #ff6347; /* Coral color */
            border-radius: 5px;
            text-align: center;
            padding: 10px 20px;
            display: block;
            margin: 0 auto;
            cursor: pointer;
        }
    </style>
    <button class="predict-button">Predict</button>
"""

# Display the button with custom CSS
st.markdown(predict_button_html, unsafe_allow_html=True)

# Prediction
if st.button("Predict", key='predict_button'):
    # Prepare input features as a list
    input_features = [[variance, skewness, curtosis, entropy]]

    # Make prediction using the loaded model
    prediction = model.predict(input_features)

    # Display prediction result
    if prediction[0] == 0:
        st.write("The Bank Note is Not Forged")
    else:
        st.write("The Bank Note is Forged")