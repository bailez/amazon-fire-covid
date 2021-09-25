import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import zscore

all_reports = os.listdir(r'data/cleaned/ms-dsei/')

reports = list()

for i in all_reports:
    
    rep = pd.read_csv(r'data/cleaned/ms-dsei/' + i)
    reports.append(rep)

df = pd.concat(reports)

confirmed = df.pivot(columns='DSEI', index = 'Data', values='Casos Confirmados').sort_index()
deceased = df.pivot(columns='DSEI', index = 'Data', values='Obitos').sort_index()


conf_agg = confirmed.sum(axis=1)

dec_agg = deceased.sum(axis=1)

# %% Confirmed Cases

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)


conf_agg.plot(label='Casos Confirmados Diários')

plt.legend()
plt.xticks(rotation=30)

plt.plot()


# %% Confirmed Cases

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)


conf_agg.plot(label='Casos Confirmados Diários')

plt.legend()
plt.xticks(rotation=30)

plt.plot()
# %% Casos Confirmados Diário
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)


conf_agg.diff().plot(label='First difference',lw=0.7)
conf_agg.diff().rolling(7).mean().plot(label='7-day rolling mean', ls='--', color='black')

plt.legend()
plt.xticks(rotation=30)

plt.plot()

# %% Obitos

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)


dec_agg.plot(label='Óbitos Diários')

plt.legend()
plt.xticks(rotation=30)

plt.plot()
# %% Casos Confirmados Diário
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)


dec_agg.diff().plot(label='First difference',lw=0.7)
dec_agg.diff().rolling(7).mean().plot(label='7-day rolling mean', ls='--', color='black')

plt.legend()
plt.xticks(rotation=30)
plt.plot()

# %%
queimada = pd.read_csv(r'data/cleaned/inpe/Focos_2021-01-01_2021-09-06.csv')

queimada['datahora'] = pd.to_datetime(queimada['datahora'])

queimada = queimada.set_index('datahora')

queimada.index.name = 'Data'

queimada = queimada[queimada['satelite'] == 'AQUA_M-T']

queimada['N'] = 1

queimada = queimada.resample('d').sum()

# %% Fire Radiation Power AQUA M - T

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)


queimada['frp'].plot(label='FRP (nivel)',lw=1)

#plt.title('', fontsize=19)
plt.legend()
plt.xticks(rotation=30)
plt.plot()

# %% Fire Radiation Power AQUA M - T LOG SCALE

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)


np.log(queimada['frp']).plot(label='FRP (Log Scale)',lw=1)


plt.legend()
plt.xticks(rotation=30)
plt.plot()


# %% Ocorrência de Queimadas AQUA M - T


fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)


queimada['N'].plot(label='Ocorrências (nível)',lw=1)

#plt.title('', fontsize=19)
plt.legend()
plt.xticks(rotation=30)
plt.plot()

# %% Ocorrência de Queimadas AQUA M - T    LOG SCALE
 

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)


np.log(queimada['N']).plot(label='Ocorrências (Log Scale)',lw=1)

#plt.title('', fontsize=19)
plt.legend()
plt.xticks(rotation=30)
plt.plot()
