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
import plotly.offline as py
import plotly.graph_objs as go

INFOS = {"inhabitants": {"Germany":80000000, "Switzerland": 6000000, "France":65000000, "Austria": 9000000, "Poland": 37000000},
         "beds": {'Andorra': 6,
                  'Austria': 1833,
                  'Belgium': 1755,
                  'Bulgaria': 913,
                  'Croatia': 650,
                  'Cyprus': 92,
                  'Czech Republic': 1227,
                  'Denmark': 372,
                  'Estonia': 196,
                  'Finland': 329,
                  'France': 7540,
                  'Germany': 23890,
                  'Greece': 680,
                  'Hungary': 1374,
                  'Iceland': 29,
                  'Ireland': 289,
                  'Italy': 7550,
                  'Latvia': 217,
                  'Lithuania': 502,
                  'Luxembourg': 127,
                  'The Netherlands': 1065,
                  'Norway': 395,
                  'Poland': 2635,
                  'Portugal': 451,
                  'Romania': 4574,
                  'Slovakia': 500,
                  'Slovenia': 131,
                  'Spain': 4479,
                  'Sweden': 550,
                  'Switzerland': 866,
                  'UK': 4114},
         "counter_measures_dates": {"Germany":dt.datetime(2020,3,14), "Switzerland": dt.datetime(2020,3,14), "France":dt.datetime(2020,3,14), "Austria": dt.datetime(2020,3,14), "Poland": dt.datetime(2020,3,14)}}

def load_data():
    df = pd.read_csv(io.StringIO(requests.get("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv").content.decode('utf-8')))
    return df

def get_beds():
    beds = pd.read_html("https://link.springer.com/article/10.1007/s00134-012-2627-8/tables/2")
    beds = beds[0]
    tmp = beds[["Â","Critical care beds"]]
    tmp.columns = ["country","beds"]
    tmp.index = tmp["country"]
    tmp = tmp[["beds"]]
    return tmp.T.to_dict("records")

def main():
    st.AppName = "Corona Beds"
    df_raw = load_data()
    st.markdown("# Corona - Wie lange reichen die Betten?")
    st.sidebar.markdown("# Infos")
    st.sidebar.markdown("### Diese App versucht vorauszuagen ob die Krankenhausbetten bei aktueller Ansteckungsquote auch in Zukunft reichen.")
    st.sidebar.markdown("Die Infektions-Daten kommen tagesaktuell von der [JHU CSSE](https://github.com/CSSEGISandData/COVID-19)")
    st.sidebar.markdown("Die Schätzung der verfügbaren Betten in Europa wurde [hier](https://link.springer.com/article/10.1007/s00134-012-2627-8) entommen. Die durchschnittliche Hospitalisierungsquote [hier](https://youtu.be/Fx11Y4xjDwA).")
    st.sidebar.markdown("Bei der Voraussage wird ein logistisches Wachstum mit Kapazitätsgrenze (totale Anzahl Infektionen) angenommen.")
    st.sidebar.markdown("Autor: plotti@gmx.net")
    country = st.selectbox("Land",tuple(INFOS["beds"].keys()),index=tuple(INFOS["beds"].keys()).index("Switzerland"))
    periods = st.slider('Voraussage für wieviele Tage?', 0, 50, 20)
    max_hospitalbeds = st.slider('Anzahl verfügbare Betten in %s ' % country, 0, int(INFOS["beds"][country]*2), int(INFOS["beds"][country]))
    percentage = st.slider('Wieviel Prozent aller Infizierten müssen ins Spital? (Hospitalisierungsquote)', 0, 20, 5)
    max_infections  = int(max_hospitalbeds / (percentage/100))
    st.markdown("Maximal mögliche Anzahl an infizierten Fälle bei %s Prozent Hospitalisierungsquote: ** %s ** " % (percentage,max_infections))
    max_cases = st.slider('Ihre Annahme: Geschätzte totale Anzahl an Infektionen in %s ' % country, max_infections, int(3*max_infections), int(max_infections*1.5))

    if st.button('Berechnung beginnen'):
        df = df_raw
        df = pd.DataFrame(df[df["Country/Region"] == country].head(1))
        df = df.T.iloc[4:].reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"],infer_datetime_format=True).dt.date
        df = df[df["y"]>0]
        df.reset_index(drop=True)
        df["cap"] = max_cases
        df['ds'] = df['ds'].astype('datetime64[ns]')
        result = predict(df,periods,max_cases)
        result = pd.merge(df,result,on="ds",how="right")[["ds","y","yhat",'yhat_lower', 'yhat_upper',"cap_y"]]
        plot_volatility(result,max_infections)
        #st.balloons()
    else:
        st.write('')

def predict(df,periods,max_cases):
    m = Prophet(mcmc_samples=50,growth='logistic').fit(df);
    future = m.make_future_dataframe(periods=periods,freq='d')
    future["cap"] = max_cases #whats the max RW we can reach
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper',"cap"]]
    return forecast

def plot_volatility(dataframe,max_infections):
    #https://community.plot.ly/t/fill-area-upper-to-lower-bound-in-continuous-error-bars/19168
    upper_bound = go.Scatter(
        name='Obere Schätzung',
        x=dataframe['ds'],
        y=dataframe['yhat_upper'] ,
        mode='lines',
        line=dict(width=0.5,
                 color="rgb(255, 188, 0)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty')
    
    lower_bound = go.Scatter(
        name='Untere Schätzung',
        x=dataframe['ds'],
        y=dataframe['yhat_lower'],
        mode='lines',
        line=dict(width=0.5, color="rgb(141, 196, 26)"),)
    
    trace1 = go.Scatter(
        mode='markers',
        name='Bestätigte Infektionen',
        marker=dict(
            color='LightSkyBlue',
            size=10,
            line=dict(
                color='MediumPurple',
                width=2
            )
        ),
        x=dataframe['ds'],
        y=dataframe['y'],
        line=dict(color='rgb(31, 119, 180)'))
    
    trace2 = go.Scatter(
        name='Vorausage der Infektionen',
        x=dataframe['ds'],
        y=dataframe['yhat'],
        mode='lines',
        line=dict(color='rgb(246, 23, 26)'))
    
    trace3 = go.Scatter(
        name='Totale Infektionen',
        x=dataframe['ds'],
        y=dataframe['cap_y'],
        mode='lines',
        line=dict(color='rgb(0, 0, 0)'))

    trace4 = go.Scatter(
        name='Kapazitätsgrenze Spitäler',
        x=dataframe['ds'],
        y=[max_infections for i in range(0,len(dataframe))],
        mode='lines',
        line=dict(color='rgb(50, 280, 40)'))

    data = [lower_bound, upper_bound,trace1, trace2,trace3,trace4]

    layout = go.Layout(
        yaxis=dict(title='Anazl Infektionen'),
        title='Voraussage über die Anzahl an Infektionen',
        hovermode = 'closest',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=10,
                         label='Letzte 10 Tage',
                         step='day',
                         stepmode='todate'),
                    dict(count=15,
                         label='Letzte 15 Tage',
                         step='day',
                         stepmode='todate'),
                    dict(count=30,
                         label='Letzte 30 Tage',
                         step='day',
                         stepmode='todate'),
                    dict(step='all')
                ])
            ),          
            rangeslider=dict(
                visible = False
            ),
            type='date'
        ),        
        showlegend = True)

    fig = go.Figure(data=data, layout=layout)
    fig.layout.update(legend=dict(x=0.1, y=.5))
    return st.plotly_chart(fig)

if __name__ == "__main__":
    main()