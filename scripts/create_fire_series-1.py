import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/ms-dsei')

aldeias = r'data/cleaned/funai/aldeias/aldeiasPoint.shp'

bioma = gpd.read_file(r'C:\Users\felip\Downloads\Biomas_250mil\lm_bioma_250.shp')

ald = gpd.read_file(aldeias)

areas = gpd.read_file(r'data/cleaned/funai/areas_dsei/areas_dsei.shp')

queimada = pd.read_csv(r'data/cleaned/inpe/2010.csv')


queimada['datahora'] = pd.to_datetime(queimada.datahora)

#queimada_set = queimada[queimada['datahora'] == '202']
queimada_set = queimada[queimada['satelite'] == 'AQUA_M-T']
#queimada_set = queimada_set[queimada_set['satelite'] == 'AQUA_M-T']['2020']


points = list(map(lambda x, y: Point(x,y), 
                  list(queimada_set['longitude']), 
                  list(queimada_set['latitude'])))

q_set = gpd.GeoDataFrame(queimada_set, geometry=points)

## separate queimadas
# %%
for i in q_set.index:
    
    point = q_set.loc[i]['geometry']
    
    if True in areas.contains(point).values:
        pass
    else:
        q_set = q_set.drop(i)
        
    

covid = pd.read_csv(r'data/cleaned/ms-dsei/'+ all_reports[-1])

df = areas.copy()

df = df.set_index('dsei')

covid = covid.set_index('DSEI')

covid.index = covid.index.map(lambda x: x.upper())

covid['letalidade'] = covid['Obitos']/covid['Casos Confirmados']

df['letalidade'] = covid['letalidade']

df.plot(column='letalidade')

df['casos'] = covid['Casos Confirmados']


# %%


ano = '2010'

fig, axs = plt.subplots(2,2,figsize=(13,12))

#for i in range(10):
    #plt.text(float(merge.longitude[i]),
 #            float(merge.latitude[i]),"{}\n{}".format(merge.name[i]),size=10)

q_set_1q = q_set[q_set['datahora'] > f'{ano}-01-01'][q_set['datahora'] < f'{ano}-03-01']
q_set_2q = q_set[q_set['datahora'] > f'{ano}-03-01'][q_set['datahora'] < f'{ano}-06-01']
q_set_3q = q_set[q_set['datahora'] > f'{ano}-06-01'][q_set['datahora'] < f'{ano}-09-01']
q_set_4q = q_set[q_set['datahora'] > f'{ano}-09-01'][q_set['datahora'] < f'{ano}-12-01']

col = True
row = 0
q = 1
for i in [q_set_1q,q_set_2q,q_set_3q,q_set_4q]:
    bioma.plot('CD_Bioma', ax= axs[row][int(col)], legend = False,lw=1,cmap='Pastel2')
    bioma.boundary.plot(lw=0.5,ax=axs[row][int(col)], color='grey')
    i.plot(alpha=0.25, ax = axs[row][int(col)], 
                               markersize = 20, 
                               edgecolor='black', color='red')
    axs[row][int(col)].set_title(ano + 'Q' + str(q), fontsize=24)
    q +=1
    if not col:
        row += 1
    col = not col
#ald.plot(alpha=0.8, ax = ax, markersize=0.2,edgecolor='red')
#q_set.iloc[49:50,:].plot(alpha=0.5, ax = ax, markersize = 100, edgecolor='black', color='red', label='Queimadas')






