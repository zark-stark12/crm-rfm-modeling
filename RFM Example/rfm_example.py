# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:16:05 2018

@author: juan.zarco

source for dataset: https://www.kaggle.com/regivm/retailtransactiondata#Retail_Data_Transactions.csv

dataset type in this example is a transactional dataset type. Below we convert this dataset to a customer level dataset
to highlight both examples.
"""

import RFM
from RFM import rfm

data = pd.read_csv('Retail_Data_Transactions.csv')
wgts = (0.1,0.25,0.65)

model.fit(data, dataset_type='transactional')

summary = model.summary_statistics()

print(summary)

fitted_data = model.get_fitted_data()