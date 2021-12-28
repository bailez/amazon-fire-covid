import requests as req
import pandas as pd
import os

dates = pd.date_range(start='01-01-2020', end='12-31-2021')
os.chdir(r'C:\Users\felip\OneDrive\Documentos\FEA\Econometria 3')
#  %%

for date in dates:
    current_date = date.strftime('%d-%m-%Y')
    
    url = 'https://saudeindigena1.websiteseguro.com/coronavirus/pdf/'\
        f'{current_date}_Boletim epidemiologico SESAI sobre COVID 19.pdf'
        
    
    response = req.get(url)
    
    if response.status_code != 200:
        continue

    with open(f'data/raw/ms-dsei/report-{current_date}.pdf', 'wb') as f:
        print(f'report-{current_date}.pdf')
        f.write(response.content)
    
# %%
all_reports = os.listdir(r'data/cleaned/ms-dsei/')

reports = list()

for i in all_reports:    
    rep = pd.read_csv(r'data/cleaned/ms-dsei/' + i)
    reports.append(rep)

df = pd.concat(reports)
df.to_csv(r'data/cleaned/dsei.csv')