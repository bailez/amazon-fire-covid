import numpy as np
import pandas as pd
from tika import parser
import os

COLUMNS = ['Casos Suspeitos', 'Casos Confirmados', 'Descartados', 'Infectados',
           'Cura Clinica', 'Obitos']

INDEX = ['Alagoas e Sergipe', 'Altamira', 'Alto Rio Juruá', 'Alto Rio Negro', 
         'Alto Rio Purus', 'Alto Rio Solimões', 'Amapá e Norte do Pará',
         'Araguaia', 'Bahia', 'Ceará', 'Cuiabá', 'Guamá-Tocantins',
         'Interior Sul', 'Kaiapó do Mato Grosso', 'Kaiapó do Pará', 'Leste de Roraima',
         'Litoral Sul', 'Manaus', 'Maranhão', 'Mato Grosso do Sul', 'Médio Rio Purus',
         'Médio Rio Solimões e Afluentes', 'Minas Gerais e Espírito Santo',
         'Parintins', 'Pernambuco', 'Porto Velho', 'Potiguara', 'Rio Tapajós',
         'Tocantins', 'Vale do Javari', 'Vilhena', 'Xavante', 'Xingu','Yanomami']


def get_raw_text(file : str) -> str:
    
    text = parser.from_file(file)['content']
    
    return text

def generate_table(text : str, index : list) -> list:

    values = text[text.find('alagoas') : text.find('total')].lower()
    
    for i in index:
        values = values.replace(i.lower(), '')
    
    values = values.split('\n')
    
    values = list(filter(lambda x: len(x) > 0, values))
    
    values = list(map(lambda x: x.split(' '), values))
    
    table = []
    
    for col in values:
        column = list(filter(lambda x: x != '', col))
        table.append(column)
  
    return table

def create_panel(table : list, 
                 index : list, 
                 columns : list) -> pd.DataFrame:
    

    df = pd.DataFrame(table)

    df.index = index    

    df.index.name = 'DSEI'

    df.columns = columns

    df = df.astype(int)
    
    return df

# %% == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ##

all_reports = os.listdir(r'data/raw/ms-dsei/')

for report in all_reports:
    
    print(report)
            
    text = get_raw_text(file = r'data/raw/ms-dsei/' + report).lower()
    
    table = generate_table(text = text, index = INDEX)
    
    try:
    
        df = create_panel(table = table, index = INDEX, columns = COLUMNS)
    
    except Exception as e:
        print(e)
        
    
    date = report[7:-4]
    
    date = pd.to_datetime(date, format= '%d-%m-%Y')
    
    df['Data'] = date
            
    df.to_csv(r'data/cleaned/ms-dsei/' + report[:-4] + '.csv')

