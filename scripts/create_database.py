import pandas as pd
import os
import numpy as np

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

covid = pd.read_csv(r'data/cleaned/dsei.csv').dropna().iloc[:,2:]

covid = covid.groupby(['Bioma','Data']).sum().reset_index()

covid.Data = pd.to_datetime(covid.Data)

cases_covid = covid.pivot(index = 'Data', 
                          columns = 'Bioma', 
                          values = 'Casos Confirmados')

death_covid = covid.pivot(index = 'Data', 
                          columns = 'Bioma', 
                          values = 'Obitos')
# %%

inpe = pd.read_csv('data/cleaned/inpe/2020.csv')
inpe_2020 = inpe[inpe['satelite'] == 'AQUA_M-T']

inpe = pd.read_csv('data/cleaned/inpe/2021.csv')
inpe_2021 = inpe[inpe['satelite'] == 'AQUA_M-T']

# %%
inpe = pd.concat([inpe_2020, inpe_2021])

inpe.datahora = pd.to_datetime(inpe.datahora)

inpe['n'] = 1

inpe = inpe.groupby(['bioma', 'datahora']).sum().reset_index()
inpe = inpe.pivot(columns = 'bioma', 
                        index = 'datahora', 
                        values = 'frp').fillna(0)
inpe = inpe[['Amazonia', 'Cerrado', 'Mata Atlantica']]
inpe = inpe.resample('d').sum()

# %%

df = pd.concat([death_covid, cases_covid, inpe], 
               axis=1,
               keys=['Obitos','Casos', 'FRP']).dropna()
    
df.to_csv(r'data/cleaned/dataset.csv')