import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from joblib import load
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix

# Title
st.title("Customer Churn App")
st.header("Exploratory Data Analysis")
st.text("Exploratory Data Analysis (EDA) is a method of analysing data that employs visual \ntechniques. It is used to discover trends, patterns, or to validate assumptions \nusing statistical summaries and graphical representations.")

dataset = 'Customer-Churn.csv'                                                  # DataFrame
new_datasets = 'cleaned_Dataset.csv'
# Function to load dataset
@st.cache(persist=True)                                       
def explore_data(data):
	df = pd.read_csv(os.path.join(dataset))
	return df

@st.cache(persist=True)
def explore_dataset(new_df):
	df = pd.read_csv(os.path.join(new_datasets))
	return df
data = explore_data(dataset)	
new_df = explore_dataset(new_datasets)

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
	st.success(data.shape[0])
elif data_dim == "Columns":
	st.markdown("**Columns Counts**")
	st.success(data.shape[1])
else:
	st.markdown("**All Dataset shape**")
	st.success(data.shape)		
	
#------------------------------------------------------ Describe Data ---------------------------------------------------
if st.checkbox("Describe Dataset"):
	st.text("Let’s get a quick summary of the dataset using the describe method")
	st.write(data.describe())	

#------------------------------------------------------------- EDA----------------------------------------------------------

col_option = st.selectbox("You can select any one option from the select box.",("Correlation Plot", "Distribtuion of Churn","Distribution of Tenure","Churn Distribution with Tenure", "Distribution of Gender","Distribution of SeniorCitizen", "Chrun distribuiton with SeniorCitizen","Distribution of PhoneService", "Churn distribuiton with PhoneService", "Distribution of PaymentMethod", "Churn distribuiton with PhoneService"))

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
	st.plotly_chart(fig)

#---------------------------------------------------- Prediction-----------------------------------------------------------------------
st.header("Prediction")

X = new_df.drop(columns = ['Churn'])
y = new_df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state=1)

#------------------------------------------------------------ Logistic Regression -----------------------------------------------------

log_model = make_pipeline(StandardScaler(),LogisticRegression())
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
accuracy = log_model.score(X_test,y_test)
lr_matrix = confusion_matrix(y_test, y_pred)

#----------------------------------------------------------------Decision Tree----------------------------------------------------------

dec_model = make_pipeline(StandardScaler(), DecisionTreeClassifier())
dec_model.fit(X_train,y_train)
y_pred1 = dec_model.predict(X_test)
accuracy1 = dec_model.score(X_test,y_test)
dec_matrix = confusion_matrix(y_test, y_pred1)

#---------------------------------------------------------------RandomForest Classifier------------------------------------------------
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier())
rf_model.fit(X_train,y_train)
y_pred2 = rf_model.predict(X_test)
accuracy2 = rf_model.score(X_test,y_test)
rf_matrix = confusion_matrix(y_test, y_pred2)

#----------------------------------------------------------------KNeighbors Classifier-------------------------------------------------
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_model.fit(X_train,y_train)
y_pred3 = knn_model.predict(X_test)
accuracy3 = knn_model.score(X_test,y_test)
knn_matrix = confusion_matrix(y_test, y_pred3)

#------------------------------------------------------------------AdaBoostClassifier-------------------------------------------------
ada_model = make_pipeline(StandardScaler(), AdaBoostClassifier())
ada_model.fit(X_train,y_train)
y_pred4 = ada_model.predict(X_test)
accuracy4 = ada_model.score(X_test,y_test)
ada_matrix = confusion_matrix(y_test, y_pred4)

#-------------------------------------------------------------GradientBoostingClassifier-----------------------------------------------
gb_model = make_pipeline(StandardScaler(), GradientBoostingClassifier())
gb_model.fit(X_train,y_train)
y_pred5 = gb_model.predict(X_test)
accuracy5 = gb_model.score(X_test,y_test)
gb_matrix = confusion_matrix(y_test, y_pred5)

#--------------------------------------------------------------------ExtraTreesClassifier------------------------------------------------
et_model = make_pipeline(StandardScaler(), ExtraTreesClassifier())
et_model.fit(X_train,y_train)
y_pred6 = et_model.predict(X_test)
accuracy6 = et_model.score(X_test,y_test)
et_matrix = confusion_matrix(y_test, y_pred6)

options = st.selectbox("You can select any one option from the select box.",("LogisticRegression","DecisionTree Classifier","RandomForest Classifier", "KNeighbors Classifier","AdaBoost Classifier","GradientBoosting Classifier", "ExtraTrees Classifier"))

if options == "LogisticRegression":
	st.write("### Logistic Regression")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(lr_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### Logistic Regression accuracy: ",accuracy)
	st.write("### Precision - ",precision_score(y_test,y_pred))
	st.write("### Recall - ",recall_score(y_test,y_pred))
	st.write("### F1 score - ",f1_score(y_test,y_pred))

if options == "DecisionTree Classifier":
	st.write("### DecisionTreeClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(dec_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### DecisionTree accuracy: ",accuracy1)
	st.write("### Precision - ",precision_score(y_test,y_pred1))
	st.write("### Recall - ",recall_score(y_test,y_pred1))
	st.write("### F1 score - ",f1_score(y_test,y_pred1))

if options == "RandomForest Classifier":
	st.write("### RandomForestClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(rf_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### RandomForest accuracy: ",accuracy2)
	st.write("### Precision - ",precision_score(y_test,y_pred2))
	st.write("### Recall - ",recall_score(y_test,y_pred2))
	st.write("### F1 score - ",f1_score(y_test,y_pred2))

if options == "KNeighbors Classifier":
	st.write("### KNeighborsClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(knn_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### KNeighbors accuracy: ",accuracy3)
	st.write("### Precision - ",precision_score(y_test,y_pred3))
	st.write("### Recall - ",recall_score(y_test,y_pred3))
	st.write("### F1 score - ",f1_score(y_test,y_pred3))	

if options == "AdaBoost Classifier":
	st.write("### AdaBoostClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(ada_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### AdaBoost accuracy: ",accuracy4)
	st.write("### Precision - ",precision_score(y_test,y_pred4))
	st.write("### Recall - ",recall_score(y_test,y_pred4))
	st.write("### F1 score - ",f1_score(y_test,y_pred4))	

if options == "GradientBoosting Classifier":
	st.write("### GradientBoostingClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(gb_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### GradientBoosting accuracy: ",accuracy5)
	st.write("### Precision - ",precision_score(y_test,y_pred5))
	st.write("### Recall - ",recall_score(y_test,y_pred5))
	st.write("### F1 score - ",f1_score(y_test,y_pred5))	

if options == "ExtraTrees Classifier":
	st.write("### ExtraTreesClassifier")
	fig, ax = plt.subplots(figsize=(3, 3))
	st.write(sns.heatmap(et_matrix , annot=True,fmt = "d",cmap='OrRd'))
	st.pyplot(fig)
	st.write("### ExtraTrees accuracy: ",accuracy6)
	st.write("### Precision - ",precision_score(y_test,y_pred6))
	st.write("### Recall - ",recall_score(y_test,y_pred6))
	st.write("### F1 score - ",f1_score(y_test,y_pred6))		