import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/inpe/')[:-1]

'''
Satelite padrão recomendado pelo INPE que inclui queimadas da tarde e manhã
'''

satelite = 'AQUA_M-T'


bdq_reports = list()

for i in all_reports:    
    print(i)
    bdq_i = pd.read_csv(r'data/cleaned/inpe/' + i)
    bdq_i = bdq_i[bdq_i['satelite'] == satelite]
    bdq_reports.append(bdq_i)

# %%   Constroi série de ocorrência de focos de fogos

freq = 'm'
  
bdq = pd.concat(bdq_reports)

bdq.datahora = pd.to_datetime(bdq.datahora)

bdq = bdq.set_index('datahora')
bdq.index.name = 'Data'

bdq['N'] = 1

# %%     Cria dataframe com biomas

df_b = pd.DataFrame()

biomas = bdq.bioma.drop_duplicates().dropna().values

for b in biomas:
    bdq_b = bdq[bdq['bioma'] == b]
    df_b[b] = bdq_b['N'].resample(freq).sum().fillna(0)


# %% Plot de focos das queimadas em nivel com media movel de 30 dias

fig = plt.figure(figsize=(12,6))


for i in df_b:
    df_b[i].plot()    


plt.legend()
plt.title('Focos de queimadas por bioma')

plt.plot()

# %% Plot de focos das queimadas em log com media movel de 30 dias

fig = plt.figure(figsize=(12,6))


for i in df_b:
    np.log(df_b[i]).plot()    


plt.legend()
plt.title('Focos de queimadas por bioma (log)')

plt.plot()

# %% Plot de Autocorrelação com resample mensal para cada um dos biomas

fig, axs = plt.subplots(3,2,figsize=(13,12))

col = True
row = 0

for i in biomas:
    title = 'Autocorrelação das queimadas para ' + i
    plot_acf(df_b[i], lags=36, ax = axs[row][int(col)], title = title)
    if not col:
        row += 1
    col = not col
     
plt.plot()

# %% Plot Sazonal com resample mensal para cada um dos biomas

fig, axs = plt.subplots(3,2,figsize=(13,16))
col = True
row = 0
for b in biomas:
    
    title = 'Sazonalidade para ' + b
    
    df_biome = np.log(df_b[[b]])
    df_biome = df_biome.reset_index()
    df_biome['Anos'] = list(map(lambda x: x.year, df_biome['Data']))
    df_biome['Mês'] = list(map(lambda x: x.month, df_biome['Data']))
    df_biome = df_biome.pivot(index='Mês', columns = 'Anos', values=b)
    df_biome.plot(legend = False, ax = axs[row][int(col)], title = title)
    
    if not col:
        row += 1
    col = not col
# %% Histograma
fig, axs = plt.subplots(3,2,figsize=(10,13))

col = True
row = 0

for i in biomas:
    title = 'Histograma do log de focos para ' + i
    #plot_acf(df_b[i].resample('m').sum(), lags=36, ax = axs[row][int(col)], title = title)
    
    np.log(df_b[i]).plot(bins=30, 
                        ax = axs[row][int(col)], 
                        kind='hist', 
                        title=title)
    axs[row][int(col)].set_ylabel('Frequência')
    if not col:
        row += 1
    col = not col
     
plt.plot()
# %%
fig, axs = plt.subplots(3,2,figsize=(10,15))

col = True
row = 0

for i in biomas:

    title = 'Lags das series em log para ' + i

    pd.plotting.lag_plot(np.log(df_b[i]),
                         ax = axs[row][int(col)])
                        #title=title)
    axs[row][int(col)].set_title(title)

    if not col:
        row += 1
    col = not col
     
plt.plot()
