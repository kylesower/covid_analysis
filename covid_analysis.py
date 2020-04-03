#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:18:42 2020

@author: kyle
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo

plt.style.use('seaborn')
use_states = True
fit_deaths = False
date_sub = 20
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
if use_states:
    confirmed = pd.read_csv(url+'time_series_covid19_confirmed_US.csv')
    deaths = pd.read_csv(url+'time_series_covid19_deaths_US.csv')
    stateconf = confirmed.groupby('Province_State').sum().iloc[:,50:].T#.drop('New York').T
    statedead = deaths.groupby('Province_State').sum().iloc[:,50:].T#.drop('New York').T
    states = pd.read_csv('statecodes.csv', index_col = 'state')
else:
    states = pd.read_csv('countrycodes.csv', index_col = 'name')
    confirmed = pd.read_csv(url+'time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv(url+'time_series_covid19_deaths_global.csv')
    stateconf = confirmed.groupby('Country/Region').sum().iloc[:,(40-date_sub):].T
    statedead = deaths.groupby('Country/Region').sum().iloc[:,(40-date_sub):].T

if fit_deaths:
    df = statedead
else:
    df = stateconf
if use_states:
    pop = pd.read_csv('statepop.csv', index_col = 'State')
else:
    pop = pd.read_csv('pop.csv',index_col = 'CountryName')

if use_states:
    outliers = ['Virgin Islands', 'Northern Mariana Islands', 'American Samoa', \
                'Rhode Island', 'New Mexico', 'New Hampshire', 'Nebraska', \
                'Louisiana', 'Hawaii', 'Grand Princess', 'Diamond Princess', \
                 'Arkansas', 'Guam']
else:
    outliers = ['Afghanistan', 'Albania', 'Andorra', 'Angola', 'Antigua and Barbuda', \
                'Argentina', 'Armenia', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', \
                'Belarus', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Brunei', \
                'Bulgaria', 'Burma', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', \
                'Central African Republic', 'Chad', 'China', 'Congo (Brazzaville)', \
                'Congo (Kinshasa)', 'Cote d\'Ivoire', 'Cyprus', 'Denmark', 'Diamond Princess', \
                'Djibouti', 'Dominica', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', \
                'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Gabon', 'Gambia', \
                'Georgia', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', \
                'Guyana', 'Haiti', 'Holy See', 'India', 'Iran', 'Jamaica', \
                'Japan', 'Kazakhstan', 'Kenya', 'Korea, South', 'Kosovo', 'Kuwait', \
                'Kyrgyzstan', 'Laos', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', \
                'Luxembourg', 'MS Zaandam', 'Madagascar', 'Malawi', 'Malaysia', \
                'Maldives', 'Mali', 'Malta', 'Mauritania', 'Monaco', 'Mongolia', \
                'Montenegro', 'Mozambique', 'Namibia', 'Nepal', 'Nicaragua', \
                'Niger', 'Nigeria', 'North Macedonia', 'Oman', 'Pakistan', 'Papua New Guinea', \
                'Paraguay', 'Peru', 'Qatar', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', \
                'Saint Lucia', 'Saint Vincent and the Grenadines', \
                'San Marino', 'Saudi Arabia', 'Seychelles', 'Sierra Leone', 'Singapore', \
                'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'Sri Lanka', \
                'Sudan', 'Suriname', 'Sweden', 'Syria', 'Taiwan*','Tanzania', 'Timor-Leste', \
                'Togo', 'Trinidad and Tobago', 'Uganda', 'United Arab Emirates', 'Uruguay', \
                'Venezuela', 'West Bank and Gaza', 'Zambia', 'Zimbabwe'
                ]
    
df = df.T.drop(outliers).T

# If you want to limit your analysis to states under a certain threshold of dead/cases,
# uncomment the section below
'''
for state in statedead.T.index:
    print(state)
    if statedead.loc['4/1/20'][state]>50:
        print(state)
        statedead = statedead.T.drop(state).T
for state in stateconf.T.index:
    print(state)
    if stateconf.loc['4/1/20'][state]>2000:
        print(state)
        stateconf = stateconf.T.drop(state).T
'''

def fit(x,a,c,d):
    return a/(1+np.exp(-c*(x-d)))

#plt.plot(stateconf)
#plt.xticks([5*i for i in range(6)])
#plt.legend()
rates = {}
for i, state in enumerate(df.T.index):
    try:
        popt, pcov = cf(fit,np.arange(20,20+len(df.T.loc[state])),df.T.loc[state],p0=[2*df.T.loc[state].max(),0.3,30+date_sub])
    except:
        print(state)
    print(state)
    y = fit(np.arange(20,20+len(df.T.loc[state])),*popt)
    
    plt.plot(df.loc[:][state],label='data')
    plt.plot(y,label='fit')
    plt.xticks([5*i for i in range(6)])
    plt.title(state)
    if fit_deaths:
        plt.ylabel('Dead')
    else:
        plt.ylabel('Confirmed COVID-19 Cases')
    plt.xlabel('Date')
    plt.legend()
    plt.figure()
    
    rates[state]=popt

if use_states:
    z = [rates[state][0]/pop.loc[state].max()*1e6 for state in rates]
    locationmode = 'USA-states'
    locations = [np.array(states.loc[state])[0] for state in rates]
    title1 = 'Estimated Max COVID-19 Cases Per Million Population'
    title2 = 'Estimated Transmission Rate'
else:
    z = [rates[state][0]/pop.loc[state].max()*1e6 for state in rates]
    locationmode = 'ISO-3'
    locations = [np.array(states.loc[state])[1] for state in rates]
    title1 = 'Estimated Max COVID-19 Cases Per Million Population'
    title2 = 'Estimated Transmission Rate'
    
fig = go.Figure(data=go.Choropleth(
                z = z,
                #autocolorscale = True,
                #colorbar = ColorBar(title = 'Infection Rates'),
                locationmode = locationmode,
                colorscale='Reds',
                locations = locations,
                colorbar_title = title1
                ))
pyo.plot(fig)
fig = go.Figure(data=go.Choropleth(
                z=[rates[state][1] for state in rates],
                #autocolorscale = True,
                #colorbar = ColorBar(title = 'Infection Rates'),
                locationmode = locationmode,
                colorscale='Reds',
                locations = locations,
                colorbar_title = title2
                ))
            
pyo.plot(fig)
