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
from statsmodels.stats.diagnostic import het_white, het_breuschpagan, acorr_ljungbox, acorr_breusch_godfrey, kstest_normal

# %% Leitura dos dados
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/inpe/')[:-1]


satelite = 'AQUA_M-T'

bdq = pd.read_csv(r'data/cleaned/inpe/bdq_AQUA.csv')

freq = 'm'

bdq.datahora = pd.to_datetime(bdq['datahora'])

bdq = bdq.set_index('datahora')
bdq.index.name = 'Data'

bdq['N'] = 1
# %% Selecao da serie

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
# %% transformacao dos dados
'''
    Series transformation Logaritmo natural
'''
df0 = df_raw[bioma]
df1 = np.log(df0)
decomp = seasonal_decompose(df1, model="additive")

df = decomp.trend.dropna()

# %% visualizacao das series transformadas
'''
    Series plot transformations
'''
fig, axs = plt.subplots(4,1,figsize=(7,14))




df1.plot(ax=axs[0], legend=False)
axs[0].set(ylabel='log scale')

decomp.trend.plot(ax=axs[1])
axs[1].set(ylabel='trend')


decomp.seasonal.plot(ax=axs[2])
axs[2].set(ylabel='seasonal')


seasonal_resid = decomp.resid.dropna().reset_index()
axs[3].set_ylim([-1.5,1.5])
seasonal_resid.plot.scatter(x='Data', y='resid',ax=axs[3])
plt.axhline(0, color='black')

# %% descrição
'''
Variable description
'''

print(df.describe().to_latex())
# %% ADF
'''
Stationarity test
'''

adf = adfuller(df)

print(adf)
# %%
'''
Autocorrelation plot
'''

fig, axs = plt.subplots(2,1,figsize=(8,12))



title = 'Grafico de autocorrelação'
plot_acf(df, lags=40, ax = axs[0], title = title)

title = 'Grafico de autocorrelação parcial'
plot_pacf(df, lags=40, ax = axs[1], title = title)
plt.plot()

    
# %% Modelo ARMA
'''
ARMA estimate
'''
pqds = [(2,0,1),
        (2,0,2),
        (4,0,2)
        ]

fig, axs = plt.subplots(3,1,figsize=(8,12))
fits = []
c = 0
for i in pqds:

    model = ARIMA(df, order = i)

    fit = model.fit()
    fits.append(fit)
    title = f'ARIMA{str(i)}'
    df.plot(ax=axs[c], label='original data', legend=True)
    fit.fittedvalues.plot(ax=axs[c], label='fitted values', legend=True)
    axs[c].set_title(title, loc='left')
    
    c = c +1

    print('====================='*5)
    print(fit.summary().as_latex())
    
    #%%
'''
Residual plots
'''

fig, axs = plt.subplots(3,1,figsize=(8,12))
c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    

    resid_p = i.resid.reset_index()
    resid_p.columns = ['Data', 'resid']
    axs[c].set_ylim([-0.1,0.1])
    


    resid_p.plot.scatter(x='Data', y='resid',ax=axs[c])
    
    
    #i.resid.plot(ax=axs[c])
    axs[c].set_title(title, loc='left')
    c = c + 1
    
    # %%
'''
Heterokedasticity test
'''

''' White's Lagrange Multiplier Test for Heteroscedasticity '''

het_white



''' Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity '''

het_breuschpagan

#%%
'''
Scatter Autocorrelation
'''

fig, axs = plt.subplots(3,1,figsize=(7,14))
c = 0
for i, j in zip(fits, pqds):
    title = f'ARIMA{str(j)}'
    axs[c].set_xlim([-0.08,.08])
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
''''Ljung-Box test of autocorrelation in residuals.'''

acorr_ljungbox(fit.resid, lags=40, return_df=True)

''' Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation. '''

acorr_breusch_godfrey(fit, nlags=40)

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
Normality of residual
'''

print(stats.normaltest(fit.resid))


'''
Normality test -> (Kolmogorov-Smirnov test) (p-value)
'''

norm_test = kstest_normal(fit.resid, dist='norm')
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


