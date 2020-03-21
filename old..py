import streamlit as st
import numpy as np
import pandas as pd
from fbprophet import Prophet
import datetime as dt
import base64
import matplotlib.pyplot as plt
import io
import requests
from matplotlib import pyplot as plt

INFOS = {"inhabitants": {"Germany":80000000, "Switzerland": 6000000, "France":65000000, "Austria": 9000000, "Poland": 37000000},
         "counter_measures_dates": {"Germany":dt.datetime(2020,3,14), "Switzerland": dt.datetime(2020,3,14), "France":dt.datetime(2020,3,14), "Austria": dt.datetime(2020,3,14), "Poland": dt.datetime(2020,3,14)}}

def load_data():
    df = pd.read_csv(io.StringIO(requests.get("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv").content.decode('utf-8')))
    return df

def main():
    df_raw = load_data()
    st.markdown("# Corona Social Measures Prediction")
    country = st.selectbox("Land",("Switzerland","Germany","Austria","France","Poland"))
    periods = st.slider('Zeitraum', 0, 50, 20)
    max_cases = st.slider('Erwartete Maximale Ansteckung innerhalb der nächsten %s Tage' % periods, 0, int(INFOS["inhabitants"][country]/2), int(INFOS["inhabitants"][country]/10))
    start_of_measures = st.date_input("Start der Isolations-Masnahmen", value=dt.datetime(2020,3,10))

    if st.button('Berechnung beginnen'):
        df = df_raw
        df = pd.DataFrame(df[df["Country/Region"] == country].head(1))
        df = df.T.iloc[4:].reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"],infer_datetime_format=True).dt.date
        df = df[df["y"]>0]
        df.reset_index(drop=True)
        df["cap"] = max_cases
        before_measures = df[df["ds"] < start_of_measures]
        periods2 = periods-(df["ds"].max()-start_of_measures).days-1
        df['ds'] = df['ds'].astype('datetime64[ns]')
        result = predict(before_measures,periods,max_cases)
        result2 = predict(df,periods2,max_cases,False)
        result = pd.merge(df,result,on="ds",how="right")[["ds","y","yhat",'yhat_lower', 'yhat_upper',]]
        result2 = pd.merge(df,result2,on="ds",how="right")[["ds","y","yhat",'yhat_lower', 'yhat_upper',]]
        plot_results(result,result2)
        #st.balloons()
    else:
        st.write('')

def plot_results(result,result2):
    st.markdown("### Rot: Verlaufsprognose vor Maßnahmen")
    st.markdown("### Grün: Verlaufsprognose nach Maßnahmen")

    result.index = result["ds"]
    result2.index = result2["ds"]
    ax = result["y"].plot(style='r+',figsize=(20,10))  
    ax = result["yhat"].plot(style="r")
    ax = result2["y"].plot(style="go")
    ax = result2["yhat"].plot(style="g")
    ax.set_ylabel(ylabel="Infections",fontsize=18)
    ax.set_xlabel(xlabel="Time",fontsize=18)
    ax.set_title(label="Difference in Taking measures",fontsize=22)
    ax.fill_between(result.index, result["yhat"], result2["yhat"],facecolor='green', alpha=0.2, interpolate=True)
    plt.tight_layout()
    st.pyplot()

def predict(df,periods,max_cases,cap=True):
    if cap:
        m = Prophet(mcmc_samples=500,growth='logistic').fit(df);
    else:
        m = Prophet(mcmc_samples=500,growth='linear').fit(df);
    future = m.make_future_dataframe(periods=periods,freq='d')
    if cap:
        future["cap"] = max_cases #whats the max RW we can reach
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return forecast

if __name__ == "__main__":
    main()