import streamlit as st
import plotly.express as px
import pandas as pd
import os

df = px.data.gapminder()
dataset = 'Customer-Churn.csv'
def explore_data(data):
	df = pd.read_csv(os.path.join(dataset))
	return df

data = explore_data(dataset)