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
from statsmodels.stats.diagnostic import het_white, het_breuschpagan, acorr_ljungbox, acorr_breusch_godfrey

# %%
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/inpe/')[:-1]


satelite = 'AQUA_M-T'

bdq = pd.read_csv(r'data/cleaned/inpe/bdq_AQUA.csv')

freq = 'm'

bdq.datahora = pd.to_datetime(bdq['datahora'])

bdq = bdq.set_index('datahora')
bdq.index.name = 'Data'

bdq['N'] = 1
# %%

'''
    Series transformation
'''

df = pd.DataFrame()

biomas = bdq.bioma.drop_duplicates().dropna().values

for b in biomas:
    bdq_b = bdq[bdq['bioma'] == b]
    df[b] = bdq_b['N'].resample(freq).sum().fillna(0)

bioma = 'Cerrado'

# %%
'''
    Series transformation Logaritmo natural
'''
df0 = df[bioma]
df1 = np.log(df)
df2 = np.log(df).diff(12).dropna()
# %%
'''
    Series plot
'''
fig, axs = plt.subplots(3,1,figsize=(8,12))



plt.title(f'Focos de queimadas para {bioma} aplicadondo ln')
    
plt.plot()
# %%

'''
    Series transformation
'''
df1 = np.log(df).diff(12).dropna()



# %% Plot de focos das queimadas em nivel com media movel de 30 dias
'''
    Series plot
'''
fig = plt.figure(figsize=(12,6))

df[bioma].plot()    

plt.title(f'Focos de queimadas para {bioma} aplicando ln e diferença de 12 meses')
    
plt.plot()

# %%
'''
Variable description
'''
print(df[bioma].describe())
# %% ADF
'''
Stationarity test
'''

adf = adfuller(df[bioma])

print(adf)
# %%
'''
Autocorrelation plot
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Grafico de autocorrelação'
plot_acf(df[bioma], lags=40, ax = ax, title = title)
plt.plot()
# %%
'''
Partial Autocorrelation
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Grafico de autocorrelação parcial'
plot_pacf(df[bioma], lags=40, ax = ax, title = title)
plt.plot()

    
# %% Modelo ARMA
'''
ARMA estimate
'''

model = ARIMA(df[bioma], order = (3,0,3))

fit = model.fit()


print(fit.summary())
#%%

print(sm.stats.durbin_watson(fit.resid.values))
# %%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = fit.resid.plot(ax=ax)

# %%
print(stats.normaltest(fit.resid))
# %%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = qqplot(fit.resid, line="q", ax=ax, fit=True)



# %%
'''
Autocorrelation of resid
'''

''''Ljung-Box test of autocorrelation in residuals.'''

acorr_ljungbox(fit.resid, lags=40, return_df=True)

''' Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation. '''

acorr_breusch_godfrey(fit, nlags=40)

# %%

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(fit.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(fit.resid, lags=40, ax=ax2)

# %%

'''
Normality test -> (Kolmogorov-Smirnov test) (p-value)
'''

norm_test = kstest_normal(fit.resid, dist='norm')

# %%

'''
Heterokedasticity test
'''

''' White's Lagrange Multiplier Test for Heteroscedasticity '''

het_white(resid, )



''' Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity '''

het_breuschpagan
