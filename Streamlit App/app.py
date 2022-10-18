import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from joblib import load
import pickle
import os


# Title
st.title("Customer Churn App")
st.header("Exploratory Data Analysis")
st.text("eee")


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
		st.text("Let's see the first five rows of dataset. The head function will be used to print the first five rows.")
		st.write(data.head())
	elif st.button("Tail"):
		st.write(data.tail())
	else:
		st.write(data.head(2))	

if st.checkbox("Show all Dataset"):
	st.dataframe(data)		


#----------------------------------------------------- Show Column Name-----------------------------------------------
if st.checkbox("Show Columns Name"):
	st.write(data.columns)	


#------------------------------------------------------ Show Dimensions ------------------------------------------------
data_dim = st.radio("Data Dimensions", ("Rows", "Columns", "All"))
if data_dim == "Rows":
	st.markdown("**Rows Counts**")
	st.write(data.shape[0])
elif data_dim == "Columns":
	st.markdown("**Columns Counts**")
	st.write(data.shape[1])
else:
	st.markdown("**All Dataset shape**")
	st.write(data.shape)		
	
#------------------------------------------------------ Describe Data ---------------------------------------------------
if st.checkbox("Describe Dataset"):
	st.text("Let’s get a quick summary of the dataset using the describe method")
	st.write(data.describe())	

#------------------------------------------------------------- EDA----------------------------------------------------------

col_option = st.selectbox("Select Plot",("Correlation Plot", "Distribtuion of Churn","Distribution of Tenure","Churn Distribution with Tenure", "Distribution of Gender","Distribution of SeniorCitizen", "Chrun distribuiton with SeniorCitizen","Distribution of PhoneService", "Churn distribuiton with PhoneService", "Distribution of PaymentMethod", "Churn distribuiton with PhoneService"))

#----------------------------------------------------------- Correlation ----------------------------------------------------

if col_option == "Correlation Plot":
            st.write("### Correlation")
            fig, ax = plt.subplots(figsize=(10,10))
            st.write(sns.heatmap(data.corr(),annot=True, vmax=.8, square=True, cmap='RdBu'))
            st.pyplot(fig)

#-------------------------------------------------------- Distribution of Churn ---------------------------------------------

if col_option == "Distribtuion of Churn":
		st.write("### Distribtuion of Churn")
		fig, ax = plt.subplots(figsize=(10,10))
		st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%", colors = ['lightblue', 'pink'],shadow = True,startangle=90))
		st.pyplot(fig)
		st.write(data.iloc[:,-1].value_counts())

#--------------------------------------------- Distribution of Tenure --------------------------------------------------------

if col_option == "Distribution of Tenure":
	st.write(" ### Distribution of Tenure")
	fig = sns.displot(data=data, x="tenure", binwidth=3)
	st.pyplot(fig)
	st.write(data['tenure'].value_counts())

#Churn yes dataset
churn_yes = pd.DataFrame(data.query('Churn == "Yes"'))

#Churn no dataset
churn_no = pd.DataFrame(data.query('Churn == "No"'))
X = pd.DataFrame(churn_yes['tenure'].value_counts().reset_index())

if col_option == "Churn Distribution with Tenure":
	st.write("### Churn Distribution with Tenure")
	fig, ax = plt.subplots(figsize=(10, 10))
	st.write(px.scatter(X, x="tenure",y='index',log_x=True, width=600))	

#----------------------------------------------  Distribution of Gender --------------------------------------------------------

if col_option == "Distribution of Gender":
	st.write(" ### Distribution of Gender")
	fig, ax = plt.subplots(figsize=(8, 7))
	st.write(sns.countplot(data=data, x="gender"))
	st.pyplot(fig)
	st.write(data['gender'].value_counts())

# ---------------------------------------------- Distribution of SeniorCitizen ---------------------------------------------------

if col_option == "Distribution of SeniorCitizen":
	st.write("### Distribution of SeniorCitizen")
	fig, ax = plt.subplots(figsize = (10, 10))
	st.write(data['SeniorCitizen'].value_counts().plot(kind='pie', colors = ['lightblue', 'darkblue'], autopct='%1.1f%%',
	        shadow = True, startangle=90))
	st.pyplot(fig)
	st.write(data['SeniorCitizen'].value_counts())

if col_option == "Chrun distribuiton with SeniorCitizen":
	st.write("### Chrun distribuiton with SeniorCitizen")
	fig, ax = plt.subplots(figsize=(10, 10))
	st.write(px.histogram(data, x="Churn", color="SeniorCitizen", barmode="group", color_discrete_map={"Yes": 'blue', "No": 'lightblue'}))
	
#--------------------------------------------------- Distribuiton with PhoneService ---------------------------------------------------

if col_option == "Distribution of PhoneService":
	st.write(" ### Distribution of PhoneService")
	fig, ax = plt.subplots(figsize=(8, 7))
	st.write(sns.countplot(data=data, x="PhoneService"))
	st.pyplot(fig)
	st.write(data['PhoneService'].value_counts())

if col_option == "Churn distribuiton with PhoneService":
	st.write("### Churn distribuiton with PhoneService")
	fig, ax = plt.subplots(figsize=(10, 10))
	st.write(px.histogram(data, x="Churn", color="PhoneService", barmode="group", color_discrete_map={"Yes": 'blue', "No": 'lightblue'}))
	st.pyplot(fig)

# -------------------------------------------------Distribution of PaymentMethod-----------------------------------------------------

#sns.catplot(data=df, x="PhoneService", kind="count");
if col_option == "Distribution of PaymentMethod":
	st.write("### Distribution of PaymentMethod")
	fig =sns.catplot(data=data, x="PhoneService", kind="count") 
	st.pyplot(fig)

if col_option =="Churn distribuiton with PhoneService":
	st.write("Churn distribuiton with PhoneService")
	# fig, ax=  plt.subplots(figsize=(10, 10))
	fig = px.histogram(data, x="Churn", color="PhoneService", barmode="group", color_discrete_map={"Yes": 'blue', "No": 'lightblue'})
	st.pyplot(fig)



#---------------------------------------------------- Prediction-----------------------------------------------------------------------
st.header("Prediction")



@st.cache
def load_model():
    return load('random_forest_model.pk')



