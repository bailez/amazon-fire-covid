#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:35:42 2021

@author: bailez
"""

import geopandas as gpd
import pandas as pd
import os
all_reports = os.listdir(r'data/cleaned/MS')
areas = gpd.read_file(r'data/cleaned/areas_dsei/areas_dsei.shp')
covid = pd.read_csv(r'data/cleaned/MS/'+ all_reports[-1])


from shapely.geometry.polygon import Polygon

point = Point(-35, -7)
polygon = areas[areas['dsei']=='ARAGUAIA']['geometry'].values[0]
print(polygon.contains(point))



# %%
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import os

all_reports = os.listdir(r'data/cleaned/MS')

areas = gpd.read_file(r'data/cleaned/areas_dsei/areas_dsei.shp')

queimada = pd.read_csv(r'data/cleaned/INPE/2021/Focos_2021-01-01_2021-09-06.csv')


queimada['datahora'] = pd.to_datetime(queimada.datahora)

queimada_set = queimada[queimada['datahora'] > '2021-09-05']
queimada_set = queimada_set[queimada_set['satelite'] == 'AQUA_M-T']


points = list(map(lambda x, y: Point(x,y), 
                  list(queimada_set['longitude']), 
                  list(queimada_set['latitude'])))

q_set = gpd.GeoDataFrame(queimada_set, geometry=points)

## separate queimadas

for i in q_set.index:
    
    point = q_set.loc[i]['geometry']
    
    if True in areas.contains(point).values:
        pass
    else:
        q_set = q_set.drop(i)
        
    

covid = pd.read_csv(r'data/cleaned/MS/'+ all_reports[-1])

df = areas.copy()

df = df.set_index('dsei')

covid = covid.set_index('DSEI')

covid.index = covid.index.map(lambda x: x.upper())

covid['letalidade'] = covid['Obitos']/covid['Casos Confirmados']

df['letalidade'] = covid['letalidade']

#df.plot(column='letalidade')

df['casos'] = covid['Casos Confirmados']


# %%
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)

title = 'Letalidade de COVID por DSEI'
plt.title(title + '\n', fontsize=19)

df.plot(column='letalidade', cmap='summer_r', ax= ax, legend = True, label='Letalidade')
q_set.plot(alpha=0.15, ax = ax, markersize = 'frp', edgecolor='black', color='red')
q_set.iloc[49:50,:].plot(alpha=0.5, ax = ax, markersize = 100, edgecolor='black', color='red', label='Queimadas')
plt.legend()


ax.set_xlabel('Longitude', fontsize = 13)
ax.set_ylabel('Latitude', fontsize = 13)









