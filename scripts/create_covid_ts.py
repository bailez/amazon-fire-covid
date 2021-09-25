import pandas as pd
import matplotlib.pyplot as plt
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

