import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def generate_dataframe(exchange_1 : pd.DataFrame, 
                       exchange_2 : pd.DataFrame,
                       exchange_1_name : str,
                       exchange_2_name : str) -> pd.DataFrame:
    
    coins = []

    for i in exchange_1:
        if i in exchange_2.columns:
            coins.append(i)
    exchange_1 = exchange_1[coins].set_index('date')
    exchange_2 = exchange_2[coins].set_index('date')
    
    exchange_1_cols = list(map(lambda x: (exchange_1_name, x), exchange_1.columns))
    exchange_2_cols = list(map(lambda x: (exchange_2_name, x), exchange_2.columns))
    
    
    exchange_1_cols = pd.MultiIndex.from_tuples(exchange_1_cols, names=["exchange", "crypto"])
    exchange_2_cols = pd.MultiIndex.from_tuples(exchange_2_cols, names=["exchange", "crypto"])
    
    
    exchange_1.columns = exchange_1_cols
    exchange_2.columns = exchange_2_cols
    
    df = pd.concat([exchange_1, exchange_2],axis=1)
    
    df = np.log(df)
    
    for i in df.columns.get_level_values('crypto'):
        sliced_coin = df.xs(i,level=1, axis=1).dropna()
        df.loc[:,(exchange_1_name,i)] = sliced_coin[exchange_1_name]
        df.loc[:,(exchange_2_name,i)] = sliced_coin[exchange_2_name]
    
    df = df.loc[df.first_valid_index():]
    
    return df


def plot_series(df : pd.DataFrame,
                exchange_1_name : str,
                exchange_2_name : str) -> None:
        
    fig, axs = plt.subplots(2,1,figsize=(10,10))
    axs[0].set_title(exchange_1_name, loc='left', fontsize=16)
    df[exchange_1_name].plot(ax=axs[0], legend=False)
    
    df[exchange_2_name].plot(ax=axs[1])
    axs[1].set_title(exchange_1_name, loc='left', fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1.5), loc='upper left', borderaxespad=0)

    return None

    
def print_description(df : pd.DataFrame,
                      exchange_1_name : str,
                      exchange_2_name : str,
                      print_latex : bool  = True )-> list:
    
    exchange_1_desc = df[exchange_1_name].describe().round(3).T
    exchange_1_desc['count'] = exchange_1_desc['count'].apply(int)
    
    
    exchange_2_desc = df[exchange_2_name].describe().round(3).T
    exchange_2_desc['count'] = exchange_2_desc['count'].apply(int)
    
    if print_latex:
        print(exchange_1_desc.to_latex())
        print(exchange_2_desc.to_latex())
    
    return exchange_1_desc,exchange_2_desc

def add_star(pvalue : float) -> str:
    
    ast = ''
    
    if pvalue < 0.1:
        ast = '*'
    
    if pvalue < 0.05:
        ast = '**'
        
    if pvalue < 0.05:
        ast = '**'
        
    if pvalue < 0.01:
        ast = '***'
            
    return str(pvalue)[:6] + ast


def adf_test(df : pd.DataFrame,
            exchange_1_name : str,
            exchange_2_name : str,
            regression : str = 'c',
            print_latex : bool  = True )-> list:
    
    adf_1 = pd.DataFrame()
    
    adf_2 = pd.DataFrame()
    
    for coin in df[exchange_1_name].columns:  
    
        adf = list(adfuller(df[exchange_1_name][coin].dropna()))
        adf = adf[:3]
        adf_1[coin] = pd.Series(adf, index = ['adf', 'pvalue', 'lag'])
        
        adf = list(adfuller(df[exchange_2_name][coin].dropna()))
        adf = adf[:3]
        adf_2[coin] = pd.Series(adf, index = ['adf', 'pvalue', 'lag'])
        
        

    adf_table_1 = adf_1.T.round(3)
    adf_table_1.pvalue = adf_table_1.pvalue.apply(add_star)
    adf_table_1.lag = adf_table_1.lag.apply(int)    
    
    adf_table_2 = adf_2.T.round(3)
    adf_table_2.pvalue = adf_table_2.pvalue.apply(add_star)
    adf_table_2.lag = adf_table_2.lag.apply(int)

    if print_latex:
        print(adf_table_1.to_latex())   
        print(adf_table_2.to_latex())
        
    return adf_table_1, adf_table_2