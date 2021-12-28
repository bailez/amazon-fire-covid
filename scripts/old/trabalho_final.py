import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

covid = pd.read_csv(r'data/cleaned/dsei.csv').dropna().iloc[:,2:]

covid = covid.groupby(['Bioma','Data']).sum().reset_index()


cases_covid = covid.pivot(index = 'Data', 
                          columns = 'Bioma', 
                          values = 'Casos Confirmados')



death_covid = covid.pivot(index = 'Data', 
                          columns = 'Bioma', 
                          values = 'Obitos')

death_total = death_covid.sum()
cases_total = cases_covid.sum()
death_rate = death_total/cases_total

# %%


dsei = gpd.read_file(r'data/cleaned/funai/areas_dsei/areas_dsei.shp')

dsei = dsei.set_index('dsei')

death_rate.index = death_rate.index.map(lambda x: x.upper())
death_total.index = death_total.index.map(lambda x: x.upper())
cases_total.index = cases_total.index.map(lambda x: x.upper())

dsei['letalidade'] =  death_rate
dsei['obitos'] =  death_total
dsei['casos'] =  cases_total
# %%
print(cases_covid.describe().round(3).to_latex())
print(death_covid.describe().round(3).to_latex())



# %%
param = cases_covid.copy()

l_b = 0
u_b = 10

sel = param.iloc[-1,:].sort_values(ascending=False)

sel = list(sel.iloc[l_b:u_b].index)

# %%


fig = plt.figure(figsize=(13, 6))
#ax = fig.add_subplot(111)

#plt.title('Casos Confirmados de Covid19 por DSEI', fontsize=19)
for i in sel:
    cases_covid[i].plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()

# %% Casos Confirmados Diário

fig = plt.figure(figsize=(13, 6))
#ax = fig.add_subplot(111)

#fgplt.title('Óbitos por Covid19 por DSEI ', fontsize=19)
for i in sel:
    death_covid[i].plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()


# %%


fig = plt.figure(figsize=(13, 6))
#ax = fig.add_subplot(111)

#plt.title('Casos Confirmados de Covid19 por DSEI (log)', fontsize=19)
for i in sel:
    np.log(cases_covid[i]).diff().plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()

# %% Casos Confirmados Diário

fig = plt.figure(figsize=(13, 6))
#ax = fig.add_subplot(111)

plt.title('Óbitos por Covid19 por DSEI (log)', fontsize=19)
for i in sel:
    np.log(death_covid[i]).diff().diff().plot(lw=2)


plt.legend()
plt.xticks(rotation=30)

plt.plot()
# %%
obit = {}
for i in biomas:
    obit.update({i : np.log(death_covid[i]).diff(6).dropna()})
# %%

'''

teste ADF first_differences

'''
adfs = []

trends_fd = obit.copy()

for i in biomas:
    adf = adfuller(trends_fd[i])

    adf_df = pd.Series(adf[:3], index = [(i,'adf'), 
                                     (i,'pvalue'),
                                     (i,'lag')])
    adf_df.index = pd.MultiIndex.from_tuples(adf_df.index)
    adfs.append(adf_df)
    
adfs = pd.DataFrame(pd.concat(adfs)).T
print(adfs.to_latex())


# %%

pd.concat([death_covid
