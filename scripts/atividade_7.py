import os
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid\scripts')
from funcs import plot_seasonalcomp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss



os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')



all_reports = os.listdir(r'data/cleaned/inpe/')[:-1]
satelite = 'AQUA_M-T'
bdq = pd.read_csv(r'data/cleaned/inpe/bdq_AQUA.csv')
freq = 'w'
bdq.datahora = pd.to_datetime(bdq['datahora'])
bdq = bdq.set_index('datahora')
bdq.index.name = 'Data'
bdq['N'] = 1

df = pd.DataFrame()

biomas = bdq.bioma.drop_duplicates().dropna().values

for b in biomas:
    bdq_b = bdq[bdq['bioma'] == b]
    df[b] = bdq_b['N'].resample(freq).sum().fillna(0)


biomas = ['Cerrado', 'Mata Atlantica', 'Amazonia']

df = df[biomas]
df.to_excel(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid\data.xlsx')
cerrado = np.log(df['Cerrado'])
mata_atlantica = np.log(df['Mata Atlantica'])
amazonia = np.log(df['Amazonia'])
# %%

cerrado_decomp = seasonal_decompose(cerrado, model="additive")
matatl_decomp = seasonal_decompose(mata_atlantica, model="additive")
amazonia_decomp = seasonal_decompose(amazonia, model="additive")
# %%

plot_seasonalcomp(cerrado,cerrado_decomp)
plot_seasonalcomp(mata_atlantica,matatl_decomp)
plot_seasonalcomp(amazonia,amazonia_decomp)
# %% descrição
'''
Variable description
'''

print(df.describe().to_latex())


# %%

'''

teste ADF
'''

adf = adfuller(df)

adf_df = pd.Series(adf[:3], index = [('original','adf'), 
                                     ('original','pvalue'),
                                     ('original','lag')])
adf_df.index = pd.MultiIndex.from_tuples(adf_df.index)
# %%


'''

Melhor modelo Enders

'''

# %%


'''

 ADF-GLS, PP e KPSS
'''

adf = kpss(df)

adf_df = pd.Series(adf[:3], index = [('original','adf'), 
                                     ('original','pvalue'),
                                     ('original','lag')])
adf_df.index = pd.MultiIndex.from_tuples(adf_df.index)

# %%

'''
Teste em primeira diferença

'''


# %%


'''
Grau de integração das séries

'''