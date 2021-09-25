import requests as req
import pandas as pd

dates = pd.date_range(start='01-01-2020', end='12-31-2021')


for date in dates:
    current_date = date.strftime('%d-%m-%Y')
    
    url = 'https://saudeindigena1.websiteseguro.com/coronavirus/pdf/'\
        f'{current_date}_Boletim epidemiologico SESAI sobre COVID 19.pdf'
        
    
    response = req.get(url)
    
    if response.status_code != 200:
        continue

    with open(f'report-{current_date}.pdf', 'wb') as f:
        print(f'report-{current_date}.pdf')
        f.write(response.content)
    
