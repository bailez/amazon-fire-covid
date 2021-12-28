import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')

covid = pd.read_csv(r'data/cleaned/dsei.csv').iloc[:,1:]

grupo = {'Cerrado' :  ['Araguaia', 'Parintins', 'Mato Grosso do Sul',
                      'Cuiabá', 'Xavante', 'Xingu', 'Tocantins'],

'Amazonia' : ['Altamira', 'Alto Rio Juruá', 'Alto Rio Negro', 
         'Alto Rio Purus', 'Alto Rio Solimões', 'Amapá e Norte do Pará',
         'Guamá-Tocantins', 'Kaiapó do Mato Grosso', 'Kaiapó do Pará', 
         'Leste de Roraima','Manaus', 'Maranhão', 'Médio Rio Purus',
         'Médio Rio Solimões e Afluentes','Porto Velho', 'Rio Tapajós',  
         'Vale do Javari', 'Vilhena', 'Yanomami'],


'Mata Atlantica' : ['Minas Gerais e Espírito Santo',
                    'Bahia',
                    'Litoral Sul',
                    'Interior Sul']}

covid['Bioma'] = ''
# %%

for i in covid.index:
    for j in grupo:
        if covid.loc[i].DSEI in grupo[j]:
            covid.loc[i, 'Bioma'] = j
    
covid.to_csv(r'data/cleaned/dsei.csv')
