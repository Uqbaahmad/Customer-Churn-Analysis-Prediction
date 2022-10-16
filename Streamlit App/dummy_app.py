from asyncore import write
import streamlit as st 

import requests 



st.title("Customer Churn EDA & Prediction App")
st.text("Built with Streamlit")


# headers and Subheader
st.header("EDA App")
st.subheader("Customer Churn Dataset")


# Check Box
if st.checkbox("Show Dataset"):
	st.text("Showng Dataset")


# Radio Buttons
gender = st.radio("What is your gender?", ("Male", "Female"))

if gender == "Male":
	st.text("Hello frnds")

#Selection
occupation = st.selectbox("Occupation", ("Programmer", "Data Scientist", "Data Analyst")) 

#Sliders
age = st.slider("Your Age", 1, 99)


# Buttons
if st.button("About Us"):
	st.text("Helllo Usss...")

# Write 
st.write("Hello World")


# Images
from PIL import Image
import datetime

#st.images()
st.date_input("Todays date",datetime.datetime.now() )

# Audio
#Vedio