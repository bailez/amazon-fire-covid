import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

all_reports = os.listdir(r'data/cleaned/ms-dsei/')

reports = list()

for i in all_reports:
    
    rep = pd.read_csv(r'data/cleaned/ms-dsei/' + i)
    reports.append(rep)

df = pd.concat(reports)


spct = df.pivot(columns='DSEI', index = 'Data', values='Casos Suspeitos').sort_index()
conf = df.pivot(columns='DSEI', index = 'Data', values='Casos Confirmados').sort_index()
discard = df.pivot(columns='DSEI', index = 'Data', values='Descartados').sort_index()
infect = df.pivot(columns='DSEI', index = 'Data', values='Infectados').sort_index()
cured = df.pivot(columns='DSEI', index = 'Data', values='Cura Clinica').sort_index()
decsd = df.pivot(columns='DSEI', index = 'Data', values='Obitos').sort_index()


sdecsd = decsd.sum(axis=1)
sconf = conf.sum(axis=1)
# %%
#fig, ax = plt.subplots(1)

#sdecsd.plot()
fig, ax = plt.subplots(figsize=(8, 6))

#ax.fill_between(df.index,df['Max'],df['Min'], alpha=0.3)


#sconf.diff().plot(label='First Differences')
#sconf.diff().rolling(window=7).mean().plot(linewidth=2,color='black', linestyle='--', label='Rolling 7 days')
np.log(qts['n']).plot(label='Log',)
plt.xticks(rotation=30)
plt.legend()


#plt.title('Risk driver: {}'.format(rd_label))

plt.show()
