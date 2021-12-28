import pandas as pd
import geopandas as gpd

r'data\cleaned\funai\areas_dsei'

queimada = pd.read_csv(r'data/cleaned/inpe/2021.csv')

queimada['datahora'] = pd.to_datetime(queimada['datahora'])

queimada = queimada.set_index('datahora')

queimada.index.name = 'Data'

queimada = queimada[queimada['satelite'] == 'AQUA_M-T']