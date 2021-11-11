# %% Importa modulos
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_white, het_breuschpagan, acorr_ljungbox, acorr_breusch_godfrey, kstest_normal, het_arch

# % Leitura dos dados
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/inpe/')[:-1]


satelite = 'AQUA_M-T'

bdq = pd.read_csv(r'data/cleaned/inpe/bdq_AQUA.csv')

freq = 'w'

bdq.datahora = pd.to_datetime(bdq['datahora'])

bdq = bdq.set_index('datahora')
bdq.index.name = 'Data'

bdq['N'] = 1
# % Selecao da serie

'''
    Series transformation
'''

df = pd.DataFrame()

biomas = bdq.bioma.drop_duplicates().dropna().values

for b in biomas:
    bdq_b = bdq[bdq['bioma'] == b]
    df[b] = bdq_b['N'].resample(freq).sum().fillna(0)

bioma = 'Cerrado'
df_raw = df.iloc[:-1,:]

# % transformacao dos dados
'''
    Series transformation Logaritmo natural
'''
df0 = df_raw[bioma]
df1 = np.log(df0)
decomp = seasonal_decompose(df1, model="additive")

df = decomp.trend.dropna()
# %%
# % visualizacao das series transformadas
'''
    Series plot transformations
'''
fig, axs = plt.subplots(4,1,figsize=(6,10))

df1.plot(ax=axs[0], legend=False)
axs[0].set(ylabel='log scale', xlabel='')


decomp.trend.plot(ax=axs[1])
axs[1].set(ylabel='trend', xlabel='')


decomp.seasonal.plot(ax=axs[2])
axs[2].set(ylabel='seasonal', xlabel='')


seasonal_resid = decomp.resid.dropna().reset_index()
axs[3].set_ylim([-1.5,1.5])
seasonal_resid.plot.scatter(x='Data', y='resid',ax=axs[3])
axs[3].set(ylabel='resid', xlabel='')
plt.axhline(0, color='black')


# %% descrição
'''
Variable description
'''

print(df.describe().to_latex())

# % ADF
'''
Stationarity test
'''

adf = adfuller(df)

adf_df = pd.Series(adf[:3], index = [('original','adf'), 
                                     ('original','pvalue'),
                                     ('original','lag')])

adf_df.index = pd.MultiIndex.from_tuples(adf_df.index)



adf = adfuller(df.diff().dropna())

adf_fd = pd.Series(adf[:3], index = [('first differences','adf'), 
                                     ('first differences','pvalue'),
                                     ('first differences','lag')])

adf_fd.index = pd.MultiIndex.from_tuples(adf_fd.index)

adf = pd.concat([adf_df.T,adf_fd.T])
adf = pd.DataFrame(adf).T
adf.index = ['']
print(adf.to_latex())

# % autocorr plot
'''
Autocorrelation plot
'''

fig, axs = plt.subplots(2,1,figsize=(8,10))

title = 'Grafico de autocorrelação'
plot_acf(df, lags=40, ax = axs[0], title = title)

title = 'Grafico de autocorrelação parcial'
plot_pacf(df, lags=40, ax = axs[1], title = title)
plt.plot()

    
# %% Modelo ARMA
'''
ARMA estimate forecast 1
'''
start_year = '2018'
prediction = '2021-04-30'
pqds = [(2,1,3),
        (3,1,15),
        (5,1,10)]

fig, axs = plt.subplots(1,1,figsize=(15,8))
#title = f'Ocorrência de queimadas no Cerrado em log e decomposto sazonalmente'
df.plot(ax=axs, legend=False)
#axs.set_title(title, loc='left')
# %%
fits = []

c = 0
fig, axs = plt.subplots(3,1,figsize=(8,12))

for i in pqds:

    model = ARIMA(df, order = i)

    fit = model.fit()
    fits.append(fit)
    title = f'    ARIMA{str(i)}'
    df[start_year:].plot(ax=axs[c], label='original data', legend=True)
    fit.fittedvalues[start_year:].plot(ax=axs[c], label='fitted values', 
                         legend=True, color='red',
                        linestyle=':')
    axs[c].set_title(title, loc='left')
    
    c = c +1

    #    print(fit.summary().as_latex())


fig, axs = plt.subplots(3,1,figsize=(8,12))

c = 0
for i, j in zip(fits,pqds):
    title = f'    ARIMA{str(j)}'
    
    fcst = i.get_forecast(prediction).summary_frame()
    
    df[start_year:].append(fcst['mean']).plot(ax=axs[c], 
                                 label='Forecast', 
                                 style='k--',                                 
                                 legend=True)
    

    axs[c].fill_between(fcst.index, 
                    fcst['mean_ci_lower'], 
                    fcst['mean_ci_upper'], 
                    color='k', alpha=0.2)
    
    df[start_year:].plot(ax=axs[c], 
                label='Original data', 
                legend=True)
    axs[c].set_title(title, loc='left')
    
    c = c +1
    
# %%
'''
Residual plots : HETEROSCEDASTITCITY
'''

fig, axs = plt.subplots(3,1,figsize=(8,14))

c = 0

for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    

    resid_p = i.resid.reset_index()
    resid_p.columns = ['Data', 'resid']
    axs[c].set_ylim([-0.1,0.1])

    resid_p.plot.scatter(x='Data', y='resid',ax=axs[c])
      
    axs[c].set_title(title, loc='left')
    c = c + 1
#%%
'''
Scatter Autocorrelation : AUTOCORRELATION
'''

fig, axs = plt.subplots(3,1,figsize=(7,14))
c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    axs[c].set_xlim([-0.04,.04])
    pd.plotting.lag_plot(fit.resid,ax=axs[c], marker='.')
    axs[c].set_title(title, loc='left')
    c = c + 1
# teste de autocorrelação no lag 1
print(sm.stats.durbin_watson(fit.resid.values))

# %% 
'''
Residual Autocorrelation
'''
       
fig, axs = plt.subplots(3,1,figsize=(8,12))

c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    plot_acf(i.resid, lags=40, ax = axs[c])
    axs[c].set_title(title, loc='left')
    c = c + 1
    
    # %% 
'''
Residual Partial-Autocorrelation
'''
       
fig, axs = plt.subplots(3,1,figsize=(8,12))

c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    plot_pacf(i.resid, lags=40, ax = axs[c])
    axs[c].set_title(title, loc='left')
    c = c + 1  
# %%

'''
ARMA estimate forecast 2
'''

N = 30


df_3 = df.iloc[:-N]


fig, axs = plt.subplots(3,1,figsize=(8,12))
fits = []
c = 0

for i in pqds:

    model = ARIMA(df_3, order = i)

    fit = model.fit()
    fits.append(fit)
    title = f'    ARIMA{str(i)}'
    df_3[start_year:].plot(ax=axs[c], label='original data', legend=True)    
    axs[c].set_title(title, loc='left')
    
    c = c +1
    print('====================='*5)
    print(fit.summary().as_latex())
fig, axs = plt.subplots(3,1,figsize=(8,12))

c = 0
for i, j in zip(fits,pqds):
    title = f'    ARIMA{str(j)}'
    
    
    fcst = i.get_forecast('2021-02-28').summary_frame()
    
    df_3[start_year:].append(fcst['mean']).plot(ax=axs[c], 
                                 label='Forecast', 
                                 style='k--',                                 
                                 legend=True)
    

    axs[c].fill_between(fcst.index, 
                    fcst['mean_ci_lower'], 
                    fcst['mean_ci_upper'], 
                    color='k', alpha=0.2)
    
    df[start_year:].plot(ax=axs[c], 
                label='Original data', 
                legend=True)
    #i.fittedvalues.plot(ax=axs[c], label='fitted values', 
     #                     legend=True, color='orange',
      #                    linestyle=':')
    axs[c].set_title(title, loc='left')
    
    c = c +1
# %%

'''
ARMA estimate forecast 3
'''

df_3 = df.iloc[:-30]


fig, axs = plt.subplots(3,1,figsize=(8,12))
fits = []
c = 0

for i in pqds:

    model = ARIMA(df_3, order = i)

    fit = model.fit()
    fits.append(fit)
    title = f'    ARIMA{str(i)}'
    df_3[start_year:].plot(ax=axs[c], label='original data', legend=True)    
    axs[c].set_title(title, loc='left')
    
    c = c +1
    # %
fig, axs = plt.subplots(3,1,figsize=(8,12))

c = 0
for i, j in zip(fits,pqds):
    title = f'    ARIMA{str(j)}'
    
    fcst = i.get_forecast(prediction).summary_frame()
    
    df_3[start_year:].append(fcst['mean']).plot(ax=axs[c], 
                                 label='Forecast', 
                                 style='k--',                                 
                                 legend=True)
    

    axs[c].fill_between(fcst.index, 
                    fcst['mean_ci_lower'], 
                    fcst['mean_ci_upper'], 
                    color='k', alpha=0.2)
    
    df[start_year:].plot(ax=axs[c], 
                label='Original data', 
                legend=True)
    #i.fittedvalues.plot(ax=axs[c], label='fitted values', 
     #                     legend=True, color='orange',
      #                    linestyle=':')
    axs[c].set_title(title, loc='left')
    
    c = c +1




    # %%
''''Ljung-Box test of autocorrelation in residuals.'''

acorr_ljungbox(fit.resid, lags=40, return_df=True)

''' Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation. '''
for i in fits:
   print(acorr_breusch_godfrey(i, nlags=1))

           
# %%
'''
Normality of residual
'''

print(stats.normaltest(fit.resid))


'''
Normality test -> (Kolmogorov-Smirnov test) (p-value)
'''
for i in fits:
    print(kstest_normal(i.resid, dist='norm'))
# %%
'''
Distribution
'''
fig, axs = plt.subplots(3,1,figsize=(8,14))

c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    qqplot(i.resid, line="q", ax=axs[c], fit=True)
    axs[c].set_ylim([-3.5,4])
    #axs[c].set_title(title, loc='left')
    c = c + 1    
    # %%
'''
Resid Distribution
'''
fig, axs = plt.subplots(3,1,figsize=(8,14))

c = 0
for i, j in zip(fits, pqds):
    
    #plt.hist(i.resid, ax=axs[c])
    title = f'Resid histogram ARIMA{str(j)}'
    i.resid.hist(ax=axs[c], bins = 50)
    axs[c].set_xlim([-0.1,0.1])
    axs[c].set_title(title)
    #axs[c].set_title(title, loc='left')
    c = c + 1

# %%
for i in fits:
    print(het_arch(i.resid))
