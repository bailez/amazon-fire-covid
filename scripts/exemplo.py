import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
# %%
''' Gerando random walk '''

dates = pd.date_range('2010','2021',freq='M')
y_t = 0

serie = []
for i in range(len(dates)):
    y_t = y_t + np.random.normal()
    serie.append(y_t)

df = pd.Series(index=dates, data=serie)
df.name = 'random'

# %%
'''
Grafico em nivel
'''

fig = plt.figure(figsize=(10, 6))
plt.title('Grafico em nivel', fontsize=19)
df.plot(lw=2)
plt.plot()

# %%
'''
Primeiras diferenças
'''

fig = plt.figure(figsize=(10, 6))
plt.title('Primeiras diferenças', fontsize=19)
df.diff().plot(lw=2)
plt.plot()

# %%
'''
Autocorrelação
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Grafico de autocorrelação'
plot_acf(df, lags=36, ax = ax, title = title)
plt.plot()

# %% Plot Sazonal 
'''
Sazonalidade
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Visualizar Sazonalidade'
df_sz = df.diff().reset_index()
df_sz['Ano'] = list(map(lambda x: x.year, df_sz['index']))
df_sz['Mês'] = list(map(lambda x: x.month, df_sz['index']))
df_sz = df_sz.pivot(index='Mês', columns='Ano', values = 'random')
df_sz.plot(ax=ax, title = title)
plt.plot()

# %% Histograma
'''
Histograma
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Histograma das primeiras diferenças'
df.diff().plot(bins=20, 
                    ax = ax, 
                    kind='hist', 
                    title=title)
ax.set_ylabel('Frequência')
plt.plot()

# %% Dispersão dos lags
'''
Dispersão dos lags
'''
fig, ax = plt.subplots(1,1,figsize=(10,6))
title = 'Dispersão dos lags'
pd.plotting.lag_plot(df.diff(), ax = ax)
ax.set_title(title)
plt.plot()
