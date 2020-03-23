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
import posthog 

posthog.api_key = "BpzsDfpzlIZiAWLvdEt-3zReZXheDKtUmLXaTkVKnTg"#'HGofPqwl6U5VgDtx_eO-M5kbH4PVBQSSd5U9g60A3Eg'

INFOS = {"inhabitants": {"Germany":80000000, "Switzerland": 6000000, "France":65000000, "Austria": 9000000, "Poland": 37000000},
         "names": {'Andorra': "Andorra", 'Austria': "Österreich", 'Belgium': "Belgien", 'Bulgaria': "Bulgarien",
                    'Croatia': "Kroatien", 'Cyprus': "Zypern", 'Czech Republic': "Tschechien", 'Denmark': "Dänemark", 'Estonia': "Estonien",
                    'Finland': "Finnland", 'France': "Frankreich", 'Germany': "Deutschland", 'Greece': "Griechenland", 'Hungary': "Ungarn",
                    'Iceland': "Island", 'Ireland': "Irland", 'Italy': "Italien", 'Lettland': "Litauen", 'Lithuania': "Littauen",
                    'Luxembourg': "Luxenburg", 'The Netherlands': "Niederlande", 'Norway': "Norwegen", 'Poland': "Polen", 'Portugal':
                    "Portugal", 'Romania': "Rumänien", 'Slovakia': "Slovakei", 'Slovenia': "Slovenien", 'Spain': "Spanien", 'Sweden':
                    "Schweden", 'Switzerland': "Schweiz", 'UK': "Vereinigtes Königreich"},
         "beds": {'Andorra': 6, 'Austria': 1833, 'Belgium': 1755, 'Bulgaria': 913, 'Croatia': 650, 'Cyprus': 92, 'Czech Republic': 1227, 
                    'Denmark': 372, 'Estonia': 196, 'Finland': 329, 'France': 7540, 'Germany': 23890, 'Greece': 680,
                    'Hungary': 1374, 'Iceland': 29, 'Ireland': 289, 'Italy': 7550, 'Latvia': 217, 'Lithuania': 502, 'Luxembourg': 127, 
                    'The Netherlands': 1065, 'Norway': 395, 'Poland': 2635, 'Portugal': 451, 'Romania': 4574, 'Slovakia': 500, 'Slovenia': 131,
                    'Spain': 4479, 'Sweden': 550, 'Switzerland': 866, 'UK': 4114},
         "counter_measures_dates": {"Germany":dt.datetime(2020,3,14), "Switzerland": dt.datetime(2020,3,14), "France":dt.datetime(2020,3,14), "Austria": dt.datetime(2020,3,14), "Poland": dt.datetime(2020,3,14)}}
reverse_countries = {v:k for k,v in INFOS["names"].items()}

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

def get_cases_to_date(df,country):
    df = pd.DataFrame(df[df["Country/Region"] == country].head(1))
    df = df.T.iloc[4:].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"],infer_datetime_format=True).dt.date
    return df.tail(1)["ds"].values[0],df.tail(1)["y"].values[0]

def main():
    posthog.capture('distinct id', 'movie played', {'movie_id': '123', 'category': 'romcom'})
    posthog.capture('distinct id', '$pageview', {'$current_url': 'https://corona-beds.herokuapp.com/'})
    st.AppName = "Corona Beds"
    df_raw = load_data()
    st.markdown("# :hospital: COVID-19 - Wie lange reichen die Betten in den Intensivstationen?")
    st.sidebar.markdown("# Infos ")
    st.sidebar.markdown(":warning: Diese App erhebt keinerlei Anspruch auf Korrektheit und basiert auf z.T. unrealistischen Annahmen.")
    st.sidebar.markdown("Diese App versucht Ihnen zu ermöglichen anhand von Daten und Annahmen vorauszusagen, ob die Betten in den Intensivstationen auch in Zukunft reichen.")
    st.sidebar.markdown("Die Infektions-Daten kommen tagesaktuell von der [JHU CSSE](https://github.com/CSSEGISandData/COVID-19).")
    st.sidebar.markdown("Die Schätzung der verfügbaren Betten in Europa stammt aus [akademischen Journals](https://link.springer.com/article/10.1007/s00134-012-2627-8). Die durchschnittliche Hospitalisierungsrate [erklärt Harald Lesch](https://youtu.be/Fx11Y4xjDwA). Die  Intensivstation-Quote wird anhand der [italienischen Daten](https://de.wikipedia.org/wiki/COVID-19-Pandemie_in_Italien#Einfluss_von_Alter%2C_Geschlecht_und_Vorerkrankungen_auf_Sterblichkeit) mit 1 Prozent geschätzt. ")
    st.sidebar.markdown("Bei der Voraussage wird ein logistisches Wachstum mit Kapazitätsgrenze angenommen (siehe Schritt 4).")
    st.sidebar.markdown("Es wird unrealistischer Weise angenommen, dass hospitalisierte Infizierte gleichmässig auf alle Intensivstationen des Landes verteilt werden können.")
    st.sidebar.markdown("Die Dunkelziffer der Infektionen kann um ein vielfaches höher als die positiv getesteten Fälle sein.")
    st.sidebar.markdown("Es wird unrealistischer Weise angenommen, dass alle Intensivstation-Betten durch Corona Patienten belegt werden.")
    st.sidebar.markdown(":warning: [staythefuckhome](https://twitter.com/hashtag/staythefuckhome)")
    st.sidebar.markdown("Autor: :blond-haired-man: plotti@gmx.net [Github](https://github.com/plotti/corona)")
    country_loc = st.selectbox("Land",tuple(INFOS["names"].values()),index=tuple(INFOS["names"].keys()).index("Switzerland"))
    country = reverse_countries[country_loc]
    max_hospitalbeds = st.slider('Schritt 1 - Passen Sie an: Wieviel verfügbare Intensivstation-Betten hat %s ?' % country_loc, 0, int(INFOS["beds"][country]*2), int(INFOS["beds"][country]))
    #periods = st.slider('Voraussage für wieviele Tage?', 0, 50, 20)
    periods = 20
    #duration = 14 # 5 days in intensive care
    percentage = st.slider('Schritt 2 - Passen Sie an: Wieviel Prozent aller Neuinfizierten müssen in %s auf die Intensivstation? ' % country_loc, 0, 5, 1)
    duration = st.slider('Schritt 3 - Passen Sie an: Wie viele Tage verbleiben Personen auf der Intensivstation? ', 2, 21, 10)
    most_current_date, cases_up_till_today = get_cases_to_date(df_raw,country)
    max_cases = st.slider('Schritt 4 - Passen Sie an: Wie viele Infektionen wird es insgesamt in %s geben? (Stand %s %s: %s)' % (country_loc,country_loc,most_current_date.strftime("%d.%m.%y"),cases_up_till_today), int(cases_up_till_today), int(4*cases_up_till_today), int(cases_up_till_today*1.5))
    if st.button('Berechnung beginnen'):
        with st.spinner('Vo­r­aus­sa­ge wird berechnet.'):
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
            plot_infections(result)
            plot_hospital_beds(result,max_hospitalbeds,duration,percentage)
    else:
        st.write('')

@st.cache(show_spinner=False)
def predict(df,periods,max_cases):
    m = Prophet(mcmc_samples=50,growth='logistic').fit(df);
    future = m.make_future_dataframe(periods=periods,freq='d')
    future["cap"] = max_cases #whats the max RW we can reach
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper',"cap"]]
    return forecast

def plot_infections(dataframe):
    for index,row in dataframe.iterrows():
        if pd.isnull(row.y):
            dataframe.loc[index,"y_merge"] = row.yhat
        else:
            dataframe.loc[index,"y_merge"] = row.y
    dataframe["day2day"] = dataframe["y_merge"].diff().rolling(3).mean()

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
        name='Vo­r­aus­sa­ge: Summe der Infektionen',
        x=dataframe['ds'],
        y=dataframe["yhat"],
        mode='lines',
        line=dict(color='rgb(246, 23, 26)'))
    
    trace3 = go.Scatter(
        name='Ihre Annahme: Summe totale Infektionen',
        x=dataframe['ds'],
        y=dataframe['cap_y'],
        mode='lines',
        line=dict(color='rgb(0, 0, 0)'))
    
    trace4 = go.Scatter(
        name='Vo­r­aus­sa­ge: Neue Infektionen pro Tag',
        x=dataframe['ds'],
        y=dataframe["day2day"],
        mode='lines',
        line=dict(color='rgb(20, 40, 120)'))

    data = [lower_bound,upper_bound,trace1,trace2,trace3,trace4]

    layout = go.Layout(
        title='Voraussage über die Anzahl an Infektionen',
        hovermode = 'closest',
        xaxis=dict(
            type='date'
        ),   
        yaxis=dict(
            title='Infizierte Personen'
        ),  
        margin=dict(l=40, r=60, t=100, b=0),
        legend=dict(orientation="h"),
        showlegend = True)
    fig = go.Figure(data=data, layout=layout)
    fig.layout.update(legend=dict(orientation="h"),
        plot_bgcolor='rgb(255,255,255)',
        separators=",.",
        dragmode=False,)
    return st.plotly_chart(fig,responsive= True)


def plot_hospital_beds(dataframe,max_hospitalbeds,duration,percentage):

    #compute day2day numbers merge prediction with history
    for index,row in dataframe.iterrows():
        if pd.isnull(row.y):
            dataframe.loc[index,"y_merge"] = row.yhat
        else:
            dataframe.loc[index,"y_merge"] = row.y
    dataframe["day2day"] = dataframe["y_merge"].diff().rolling(2).mean()*(percentage/100)
    dataframe["day2daysum"] = dataframe["y_merge"].diff().rolling(duration).sum()*(percentage/100)

    #https://community.plot.ly/t/fill-area-upper-to-lower-bound-in-continuous-error-bars/19168

    trace4 = go.Scatter(
        name='Vo­r­aus­sa­ge: Neue Patienten pro Tag, die auf die Intensivstation müssen.',
        x=dataframe['ds'],
        y=dataframe["day2day"],
        mode='lines',
        line=dict(color='rgb(20, 40, 120)'))

    trace5 = go.Scatter(
        name='Vo­r­aus­sa­ge: Summe belegter Betten durch Patienten die mind. %s Tage auf der Intensivstation bleiben müssen.' % duration,
        x=dataframe['ds'],
        y=dataframe["day2daysum"],
        mode='lines',
        line=dict(color='rgb(170, 120, 120)'))

    trace6 = go.Scatter(
        name='Anzahl total verfügbarer Betten auf Intensivstationen',
        x=dataframe['ds'],
        y=[max_hospitalbeds for i in range(0,len(dataframe))],
        mode='lines',
        line=dict(color='rgb(50, 280, 40)'))

    data = [trace4,trace5,trace6]

    layout = go.Layout(
        title='Voraussage über die Anzahl Patienten auf Intensivstationen',
        hovermode = 'closest',
        xaxis=dict(
            type='date'
        ),   
        yaxis=dict(
            title='Personen auf Intensivstation'
        ),  
        margin=dict(l=40, r=60, t=100, b=0),
        legend=dict(orientation="h"),
        showlegend = True)

    fig = go.Figure(data=data, layout=layout)
    fig.layout.update(legend=dict(orientation="h"),
        plot_bgcolor='rgb(255,255,255)',
        separators=",.",
        dragmode=False,)
    return st.plotly_chart(fig,responsive= True)

if __name__ == "__main__":
    main()