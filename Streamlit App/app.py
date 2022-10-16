import streamlit as st 


# EDa 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image



# Title
st.title("Customer Churn EDA & Prediction App")
st.header("Built with Streamlit")


# Data Frame
dataset = 'Customer-Churn.csv'

# Function to load dataset
@st.cache(persist=True)
def explore_data(data):
	df = pd.read_csv(os.path.join(dataset))
	return df

data = explore_data(dataset)

if st.checkbox("Preview Dataset"):
	if st.button("Head"):
		st.write(data.head())
	elif st.button("Tail"):
		st.write(data.tail())
	else:
		st.write(data.head(2))	

if st.checkbox("Show all Dataset"):
	st.dataframe(data)		


# Show Column Name
if st.checkbox("Show Columns Name"):
	st.write(data.columns)	


# Show Dimensions
data_dim = st.radio("Data Dimensions", ("Rows", "Columns", "All"))
if data_dim == "Rows":
	st.text("Rows Count")
	st.write(data.shape[0])
elif data_dim == "Columns":
	st.text("Columns Counts")
	st.write(data.shape[1])
else:
	st.text("All Dataset shape")
	st.write(data.shape)		
	
# Describe Data 
if st.checkbox("Describe Dataset"):
	st.write(data.describe())	

# Data Info
if st.checkbox("Info Dataset"):
	st.write(data.info())	



# Select a Columns
col_option = st.selectbox("Select Column", ("customerID", 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', "MultileLines", 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'streamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'))
if col_option == 'customerID':
	st.write(data['customerID'])
elif col_option == 'gender':
	st.write(data['gender'])
elif col_option == 'SeniorCitizen':
	st.write(data['SeniorCitizen'])
elif col_option == 'Partner':
	st.write(data['Partner'])
elif col_option == 'Dependents':
	st.write(data['Dependents'])
elif col_option == 'tenure':
	st.write(data['tenure'])
elif col_option == 'PhoneService':
	st.write(data['PhoneService'])
elif col_option == 'MultileLines':
	st.write(data['MultileLines'])				
elif col_option == 'InternetService':
	st.write(data['InternetService'])
elif col_option == 'OnlineSecurity':
	st.write(data['OnlineSecurity'])
elif col_option == 'OnlineBackup':
	st.write(data['OnlineBackup'])
elif col_option == 'DeviceProtection':
	st.write(data['DeviceProtection'])
elif col_option == 'TechSupport':
	st.write(data['TechSupport'])
elif col_option == 'StreamingTV':
	st.write(data['StreamingTV'])
elif col_option == 'streamingMovies':
	st.write(data['streamingMovies'])
elif col_option == 'Contract':
	st.write(data['Contract'])	
elif col_option == 'PaperlessBilling':
	st.write(data['PaperlessBilling'])
elif col_option == 'PaymentMethod':
	st.write(data['PaymentMethod'])
elif col_option == 'MonthlyCharges':
	st.write(data['MonthlyCharges'])
elif col_option == 'TotalCharges':
	st.write(data['TotalCharges'])
elif col_option == 'Churn':
	st.write(data['Churn'])	
else:
	st.write("Select Column")


# EDA
if st.checkbox("Correlation"):
	st.write(plt.matshow(data.corr()))
	st.pyplot()

#sns.catplot(data=df, x="Churn", kind="count");


def barPlot():
    fig = plt.figure(figsize=(10, 4))
    sns.catplot( data=data, x="Churn", kind="count")
    st.pyplot(fig)	

if st.checkbox("Seaborn Pairplot",value=True):
	fig = plt.figure()
	sns.pairplot(data, hue="Churn") 
	st.pyplot(fig)

if st.checkbox("SeniorCitizen", value=True):
	fig1 = data['SeniorCitizen'].value_counts().plot(kind='pie',
                                        colors = ['lightblue', 'darkblue'],
                                        figsize=(8,6),
                                        autopct='%1.1f%%',
                                        shadow = True,
                                        startangle=90)
	st.pyplot(fig1)									
