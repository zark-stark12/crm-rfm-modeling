# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:16:05 2018

@author: juan.zarco

source for dataset: https://www.kaggle.com/regivm/retailtransactiondata#Retail_Data_Transactions.csv

dataset type in this example is a transactional dataset type. Below we convert this dataset to a customer level dataset
to highlight both examples.
"""

import pandas as pd
import RFM_Model as rfm
from datetime import datetime

data = pd.read_csv('Retail_Data_Transactions.csv')
wgts = (0.1,0.25,0.65)
#Use these columns to read the results from the dataframe. This method will be used to print the distribution of the scoring results
def print_results(data):
    result_cols = ['Recency_Scores','Frequency_Scores','Monetary_Scores']
    results = []
    for col in result_cols:
        results.append(data[col].value_counts().divide(sum(data[col].value_counts())))
    result_df = pd.concat(results,axis=1)
    result_df.columns = result_cols
    result_df.index = [1,2,3,4,5]
    print(result_df)
    
rfm = rfm.RFM(weights=wgts)
#Default scoring method is "Quintile"
rfm.fit(data=data,dataset_type='transactional')
#the RFM_model package adds to the data copied within the class object itself and adds scores to the users based on the parameters inputted.
results = rfm.data
#print the distribution of the scoring results
print_results(results)
#Example of using different scoring methods
rfm.fit(data=data,dataset_type='transactional',scoring_method='Mean')
results = rfm.data
#print the distribution of the scoring results
print_results(results)

rfm.fit(data=data,dataset_type='transactional',scoring_method='Median')
results = rfm.data
#print the distribution of the scoring results
print_results(results)
#Using multiple scoring methods at once. Note that Recency is not stated so 'Quintile' method will be used to score Recency.
rfm.fit(data=data,dataset_type='transactional',frequency_scoring_method='Mean',monetary_scoring_method='Median')
results = rfm.data
#print the distribution of the scoring results
print_results(results)

def convert_transaction_to_user(df):
    id_col = df.columns[0]
    date_col = df.columns[1]
    transaction_col = df.columns[2]
    df[date_col] = pd.to_datetime(df[date_col])
    
    today_date = datetime.today()
    
    df_recency = df.groupby([id_col])[date_col].max().apply(lambda date: (today_date - date).days).astype(float)
    df_frequency = df.groupby([id_col])[date_col].count().astype(float)
    df_monetary = df.groupby([id_col])[transaction_col].sum().astype(float)
    full_df = pd.concat([df_recency,df_frequency,df_monetary],axis=1)
    full_df.columns = ['Recency','Frequency','Monetary']
    return full_df
#converting transactional data to customer level data to showcase generic example of applying the 'customer' parameter.
customer_data = convert_transaction_to_user(data)

rfm.fit(data=customer_data,dataset_type='customer')
results = rfm.data
#print the distribution of the scoring results
print_results(results)

#Print the dataframe result head
print(rfm.data.head())
