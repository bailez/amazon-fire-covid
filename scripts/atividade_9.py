import os
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid')
from scripts.funcs import plot_seasonalcomp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests



df = pd.read_excel(r'data.xlsx')
df = df.set_index('Data')
biomas = ['Cerrado', 'Mata Atlantica', 'Amazonia']

df = df[biomas]
df = np.log(df)

cerrado = np.log(df['Cerrado'])
mata_atlantica = np.log(df['Mata Atlantica'])
amazonia = np.log(df['Amazonia'])
# %%
decomp = {}

for i in biomas:
    decomp.update({i: seasonal_decompose(df[i])})
    
# %%
'''
Plot seasonal decomp
'''
trends = {}  
for i in decomp:    
    trends.update({i : decomp[i].trend.dropna()})

data = pd.DataFrame(trends)
df = data.dropna()
dff = df.diff().dropna()
# %%%
    
#---------------------------------------
# fitting the order of the VAR
#--------------------------------------

# To select the right order of the VAR model, we iteratively fit increasing orders 
# of VAR model and pick the order that gives a model with least AIC.
orders = pd.DataFrame(columns = ['AIC', 'BIC', 'FPE', 'HQIC'])
model = VAR(dff)
for i in range(0,13):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    orders.loc[i] = [round(result.aic,3) ,
                     round(result.bic,3),
                     result.fpe,
                     round(result.hqic,3)]


orders.index.name = "Order"
    
# In the above output, the AIC drops to lowest at lag 4, then increases at 
# lag 5 and then continuously drops further. (more negative = 'smaller' AIC)


print(orders.to_latex())

x = model.select_order(maxlags=12)

model_fitted = model.fit(3)
#print(model_fitted.summary())
# %%
#---------------------------------------
# check for remaining serial correlation
#---------------------------------------


# Serial correlation of residuals is used to check if there is any leftover pattern 
# in the residuals (errors). If there is any correlation left in the residuals, then,
# there is some pattern in the time series that is still left to be explained by the
# model. In that case, the typical course of action is to either increase the order
# of the model or induce more predictors into the system or look for a different 
# algorithm to model the time series.

# A common way of checking for serial correlation of errors can be measured using 
# the Durbin Watson’s Statistic.

# The value of this statistic can vary between 0 and 4. The closer it is to the value 
# 2, then there is no significant serial correlation. The closer to 0, there is a 
# positive serial correlation, and the closer it is to 4 implies negative serial 
# correlation.

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

# for col, val in zip(df.columns, out):
#    print(adjust(col), ':', round(val, 2))
    
for col, val in zip(df.columns, out):
    print(col, ':', round(val, 4))

dw = pd.DataFrame(out).T

dw.columns = df.columns
print(dw.to_latex())



  
#--------# %%------------------------------
# forecasting
#--------------------------------------

# In order to forecast, the VAR model expects up to the lag order number of 
# observations from the past data. This is because, the terms in the VAR model 
# are essentially the lags of the various time series in the dataset, so you 
# need to provide it as many of the previous values as indicated by the lag order
# used by the model.

# Get the lag order (we already know this)
lag_order = model_fitted.k_ar
print(lag_order)  #> 4
nobs = 52

# Input data for forecasting
forecast_input = df.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs) # nobs defined at top of program
fcst = pd.DataFrame(fc, columns = df.columns)
dates = pd.date_range(df.index[-1], freq='w', periods=nobs + 1)[1:]
fcst.index = dates
fcst.index.name = 'Data'



fig, axs = plt.subplots(1,1,figsize=(12,8))

df['2008':].plot(ax=axs)
fcst['2008':].plot(ax=axs, ls='--', color='red', legend=False)
