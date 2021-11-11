import os
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid\scripts')
from funcs import plot_seasonalcomp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss



os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')
'''
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
'''
df = pd.read_excel(r'amazon-fire-covid\data.xlsx')
df = df.set_index('Data')
biomas = ['Cerrado', 'Mata Atlantica', 'Amazonia']

df = df[biomas]
df = np.log(df)
#df.to_excel(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3\amazon-fire-covid\data.xlsx')
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
    plot_seasonalcomp(df[i],decomp[i])
    trends.update({i : decomp[i].trend.dropna()})

# %% descrição
'''
Variable description
'''
for i in biomas:
    print('\n\n', i,'\n')
    print(trends[i].describe().to_latex())

fig, axs = plt.subplots(3,1,figsize=(8,12))
c= 0
for i in biomas:
    trends[i].plot(ax=axs[c], legend=True, label=i)
    c += 1
# %%

'''

teste ADF

'''
adfs = []

for i in biomas:
    adf = adfuller(trends[i])

    adf_df = pd.Series(adf[:3], index = [(i,'adf'), 
                                     (i,'pvalue'),
                                     (i,'lag')])
    adf_df.index = pd.MultiIndex.from_tuples(adf_df.index)
    adfs.append(adf_df)
    
adfs = pd.DataFrame(pd.concat(adfs)).T
print(adfs.to_latex())

# %%

'''

 KPSS
 
'''
kpss_list = []

for i in biomas:
    kpss_test = kpss(trends[i])

    kpss_df = pd.Series(kpss_test[:3], index = [(i,'adf'), 
                                     (i,'pvalue'),
                                     (i,'lag')])
    
    kpss_df.index = pd.MultiIndex.from_tuples(kpss_df.index)
    kpss_list.append(kpss_df)
    
kpss_list = pd.DataFrame(pd.concat(kpss_list)).T
print(kpss_list.to_latex())


# %%


'''
Variable description first differences
'''
trends_fd = {}
for i in decomp:    
    plot_seasonalcomp(df[i],decomp[i])
    trends_fd.update({i : decomp[i].trend.diff().dropna()})

for i in biomas:
    print('\n\n', i,'\n')
    print(trends_fd[i].describe().to_latex())

fig, axs = plt.subplots(3,1,figsize=(8,12))
c= 0
for i in biomas:
    trends_fd[i].plot(ax=axs[c], legend=True, label=i)
    c += 1
# %%


'''

teste ADF first_differences

'''
adfs = []

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

'''

teste KPSS first_differences

'''

kpss_list = []

for i in biomas:
    kpss_test = kpss(trends_fd[i])

    kpss_df = pd.Series(kpss_test[:3], index = [(i,'adf'), 
                                     (i,'pvalue'),
                                     (i,'lag')])
    
    kpss_df.index = pd.MultiIndex.from_tuples(kpss_df.index)
    kpss_list.append(kpss_df)
    
kpss_list = pd.DataFrame(pd.concat(kpss_list)).T
print(kpss_list.to_latex())
