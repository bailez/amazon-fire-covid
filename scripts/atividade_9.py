import os
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid')
from scripts.funcs import plot_seasonalcomp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss



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
data_ = data.diff()
# %%%
