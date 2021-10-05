import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  
import os


os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

all_reports = os.listdir(r'data/cleaned/ms-dsei')

aldeias = r'data/cleaned/funai/aldeias/aldeiasPoint.shp'

bioma = gpd.read_file(r'C:\Users\felip\Downloads\Biomas_250mil\lm_bioma_250.shp')

ald = gpd.read_file(aldeias)

areas = gpd.read_file(r'data/cleaned/funai/areas_dsei/areas_dsei.shp')
    
    # %%
def fogo(ano):


    queimada = pd.read_csv(r'data/cleaned/inpe/' + str(ano) + '.csv')


    queimada['datahora'] = pd.to_datetime(queimada.datahora)


    queimada_set = queimada[queimada['satelite'] == 'AQUA_M-T']
    
    if len(queimada_set) == 0:
        continue



    points = list(map(lambda x, y: Point(x,y), 
                  list(queimada_set['longitude']), 
                      list(queimada_set['latitude'])))
    
    q_set = gpd.GeoDataFrame(queimada_set, geometry=points)
    
    fig, axs = plt.subplots(2,2,figsize=(13,12))
    
    q_set_1q = q_set[q_set['datahora'] > f'{ano}-01-01'][q_set['datahora'] < f'{ano}-03-01']
    
    q_set_2q = q_set[q_set['datahora'] > f'{ano}-03-01'][q_set['datahora'] < f'{ano}-06-01']
    q_set_3q = q_set[q_set['datahora'] > f'{ano}-06-01'][q_set['datahora'] < f'{ano}-09-01']
    q_set_4q = q_set[q_set['datahora'] > f'{ano}-09-01'][q_set['datahora'] < f'{ano}-12-01']
    
    if len(q_set_1q) == 0:
        continue
    
    col = True
    row = 0
    q = 1
    for i in [q_set_1q,q_set_2q,q_set_3q,q_set_4q]:
        
        bioma.plot('CD_Bioma', ax = axs[row][int(col)], legend = False,lw=1,cmap='Pastel2')
        bioma.boundary.plot(lw=0.5, ax = axs[row][int(col)] , color='grey')
        
        i.sample(2000).plot(alpha=0.15, ax = axs[row][int(col)],
                                   markersize = 20, 
                                   edgecolor='black', color='red')
        
        axs[row][int(col)].set_title(f'{ano}Q' + str(q), fontsize = 17)
        q += 1
        if not col:
            row += 1
        col = not col
    #ald.plot(alpha=0.8, ax = ax, markersize=0.2,edgecolor='red')
    #q_set.iloc[49:50,:].plot(alpha=0.5, ax = ax, markersize = 100, edgecolor='black', color='red', label='Queimadas')
    plt.legend()
    plt.plot()

    


# %%



for ano in range(2003,2022):

    
    queimada = pd.read_csv(r'data/cleaned/inpe/' + str(ano) + '.csv')
    
    queimada['datahora'] = pd.to_datetime(queimada.datahora)
    
    queimada_set = queimada[queimada['satelite'] == 'AQUA_M-T']
    
    points = list(map(lambda x, y: Point(x,y), 
                  list(queimada_set['longitude']), 
                      list(queimada_set['latitude'])))
    
    q_set = gpd.GeoDataFrame(queimada_set, geometry=points)
    
    if len(queimada_set) == 0:
        continue
    
    
    
    
    
    for mes in range(1,13):    
        

        fig, ax = plt.subplots(1,1,figsize=(13,13))
        bioma.plot('CD_Bioma', ax = ax, legend = False,lw=1,cmap='Pastel2')
        bioma.boundary.plot(lw=0.8, ax = ax , color='grey')
        
        q_set2 = q_set[q_set['datahora'] > f'{ano}-{mes}-01'][q_set['datahora'] < f'{ano}-{mes}-27']
        if len(q_set2) == 0:
            continue
        q_set2.plot(alpha=0.05, ax = ax,
                    markersize = 20, 
                    edgecolor='black', color='red')
        
            #break
        plt.title(f'{ano}-{mes}',fontsize=32)
        


# %%
anim


