# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:05:05 2021

@author: flpma
"""
import pandas as pd
x = r'D:\FEA\Econometria 3\dados\INPE\2020\Focos_2020-01-01_2020-12-31.csv'
df = pd.read_csv(x)

import camelot
  

report = f'D:\FEA\Econometria 3\dados\MS\report-01-07-2020.pdf'


import camelot
  
# extract all the tables in the PDF file
abc = camelot.read_pdf(report)