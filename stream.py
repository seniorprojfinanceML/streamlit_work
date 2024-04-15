import streamlit as st
from functions import Evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from datetime import datetime, timedelta, date, timezone, time
from sklearn.metrics import mean_squared_error
from io import StringIO

available_tables = ['crypto_ind_one', 'crypto_ind_adausdt', 'crypto_ind_ethusdt']
default_value_table = available_tables.index('crypto_ind_one')
selected_table = st.selectbox('Choose a table', available_tables, index=default_value_table)

default_start_date = datetime.today() - timedelta(days=15)
selected_start_date = st.date_input("Choose a start date", default_start_date)
default_start_time = st.time_input('Choose a start time', time(0,0))

default_end_date = datetime.today() - timedelta(days=1)
selected_end_date = st.date_input("Choose an end date", default_end_date)
default_end_time = st.time_input('Choose an end time', time(0,0))

available_models = ["CBR", "Bayesian", "ElasticNet", "Huber", "Lasso", "LassoLars", "Ridge", "XGB"]
default_value_model = available_models.index('CBR')
selected_model = st.selectbox("Choose a model:", available_models, index=default_value_model)

# try:
table = selected_table
start_date = date(year=selected_start_date.year, month=selected_start_date.month, day=selected_start_date.day)
start_datetime = datetime.combine(start_date, default_start_time, tzinfo=timezone.utc)
end_date = datetime(year=selected_end_date.year, month=selected_end_date.month, day=selected_end_date.day)
end_datetime = datetime.combine(end_date, default_end_time, tzinfo=timezone.utc)

# analysis = Evaluation(table=table, startDate=startDate, model=selected_model, alias = "production")
analysis = Evaluation(table=table, startDate=start_datetime, endDate=end_datetime, model=selected_model, alias="production")

st.title('Visualization of Predicted Values Over Time')

df = analysis.df
pred = analysis.pred
actual = analysis.actual
x = list(df["time"][1440:])

start_time = f"Start Time: {df['time'].iloc[0]}"
st.text(start_time)
end_time = f"End Time: {df['time'].iloc[-1]}"
st.text(end_time)

# Plotting the first line graph
fig, ax = plt.subplots()
ax.plot(x, pred)
plt.xticks(rotation=45)
# plt.ylim(-0.01, 0.01)  # Setting custom y-axis limits
ax.set_xlabel('Time')
ax.set_ylabel('Predicted Growth')
ax.set_title('Predicted Growth Over Time')
st.pyplot(fig)

# Plotting the second line graph
fig, ax = plt.subplots()
ax.plot(x, actual, label='Actual growth')
ax.plot(x, pred, label='Predicted growth')
ax.set_xlabel('Time')
plt.xticks(rotation=45)
ax.set_ylabel('Growth')
ax.set_title('Predicted Growth VS Actual Growth Over Time')
ax.legend()
st.pyplot(fig)

report = analysis.classification_report()
st.text(report)

mse = f"mse: {mean_squared_error(analysis.actual, analysis.pred)}"
st.text(str(mse))

# except Exception as e:
#     print(f"Error: {e}")
#     st.text(f"Error: {e}")