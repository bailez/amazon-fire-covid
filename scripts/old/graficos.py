import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import zscore

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')
all_reports = os.listdir(r'data/cleaned/ms-dsei/')

reports = list()

for i in all_reports:
    
    rep = pd.read_csv(r'data/cleaned/ms-dsei/' + i)
    reports.append(rep)

df = pd.concat(reports)

confirmed = df.pivot(columns='DSEI', index = 'Data', values='Casos Confirmados').sort_index()

deceased = df.pivot(columns='DSEI', index = 'Data', values='Obitos').sort_index()

letalidade = deceased/confirmed


conf_agg = confirmed.sum(axis=1)

dec_agg = deceased.sum(axis=1)

let_agg = conf_agg/dec_agg
# %%

param = confirmed.copy()

N = 10

sel = param.iloc[-1,:].sort_values(ascending=False)

sel = list(sel.iloc[:N].index)




# %% Casos Confirmados Diário

fig = plt.figure(figsize=(14, 7))
#ax = fig.add_subplot(111)

plt.title('Casos Confirmados de Covid19 por DSEI', fontsize=19)
for i in sel:
    confirmed[i].plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()
# %% Casos Confirmados Diário TRANSFORMAÇÕES

fig = plt.figure(figsize=(14, 7))


plt.title('Casos Confirmados de Covid19 por DSEI', fontsize=19)
for i in sel:
    confirmed[i].diff().plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()


# %% Obitos

fig = plt.figure(figsize=(14, 7))
#ax = fig.add_subplot(111)

plt.title('Obitos de Covid19 por DSEI', fontsize=19)
for i in sel:
    deceased[i].plot(lw=2)

#conf_agg.diff().rolling(7).mean().plot(label='7-day rolling mean', ls='--', color='black')

plt.legend()
plt.xticks(rotation=30)

plt.plot()

# %% Letalidade

fig = plt.figure(figsize=(14, 7))
#ax = fig.add_subplot(111)

plt.title('Letalidade de Covid19 por DSEI', fontsize=19)
for i in sel:
    letalidade[i].plot(lw=2)

#conf_agg.diff().rolling(7).mean().plot(label='7-day rolling mean', ls='--', color='black')

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
queimada = pd.read_csv(r'data/cleaned/inpe/2021.csv')

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

# %%