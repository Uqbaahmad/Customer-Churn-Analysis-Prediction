import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

# --------------------------------------------------------Data Info--------------------------------------------------------
if st.checkbox("Info Dataset"):
	st.text("Let’s see the columns name and their data types")
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


#------------------------------------------------------------- EDA----------------------------------------------------------

col_option = st.selectbox("Select Plot",("Correlation Plot", "Distribtuion of Churn","Distribution of Tenure","Churn Distribution with Tenure", "Distribution of Gender","Distribution of SeniorCitizen", "Chrun distribuiton with SeniorCitizen","Distribution of PhoneService", "Chrun distribuiton with PhoneService", "Distribution of PaymentMethod"))

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
	fig, ax = plt.subplots(figsize=(8, 7))
	st.write(sns.displot(data=data, x="tenure", binwidth=3))
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

if col_option == "Chrun distribuiton with PhoneService":
	st.write("### Chrun distribuiton with PhoneService")
	fig, ax = plt.subplots(figsize=(10, 10))
	st.write(px.histogram(data, x="Churn", color="PhoneService", barmode="group", color_discrete_map={"Yes": 'blue', "No": 'lightblue'}))


# -------------------------------------------------Distribution of PaymentMethod-----------------------------------------------------

st.header("Prediction")