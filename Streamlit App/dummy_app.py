import streamlit as st
import plotly.express as px
import pandas as pd
import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.markdown("Happpppppyyyyyyy   **EDA**")

dataset = 'Customer-Churn.csv'
@st.cache(persist=True)
def explore_data(data):
	df = pd.read_csv(os.path.join(dataset))
	return df

data = explore_data(dataset)
if st.checkbox("Show Pie Chart and Value Counts of Target Columns"):
        st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
        st.pyplot()
        st.write(data.iloc[:,-1].value_counts())