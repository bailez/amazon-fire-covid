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


# %%


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

print(orders.to_latex())

x = model.select_order(maxlags=12)

model_fitted = model.fit(3)
# %%




maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
   
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

granger_test = grangers_causation_matrix(df, variables = df.columns) 
print(granger_test.to_latex())
# %%



