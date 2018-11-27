# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:25:52 2018

@author: juan.zarco
"""
from copy import deepcopy
import pandas as pd
import numpy as np
import sys
from datetime import datetime

class RFM:
    '''
    RFM object to construct and build RFM model on  transactional data written in Python 3.X
    
    RFM "fit" method works with Pandas DataFrame however columns should be labeled accordingly indicating "Recency", "Frequency",
    or "Monetary for a "customer/user" dataset. If the dataset is "transactional" then the columns should be in order by customer_id, 
    order date then transaction amount. Note that "user" dataset should have customer_id as index for pandas dataframe otherwise
    it will read the default index.
    
    parameters:
        
        weights - weights corresponding to the R-F-M scores stored as a tuple;
        
        "fit" method - 
            dataset_type - "customer" or "transactional"; 
            
            "user" dataset should be labeled row by row for each individual user or consumer. Three columns should be listed 
            containing: Number of days since last transaction, Number of Transactions occurred, and the total monetary value 
            of the transactions.
            
            "transactional" dataset should be a log of all transctions labeled by a User Identification. Two columns should be
            listed containing: date of each transactions and the monetary value of each transaction.
            
            for model fitting: scoring_method - "Mean", "Median", or "Quintile";
    '''
    
    def __init__(self,weights=(0.2,0.2,0.6)):
        if sum(weights) == 1:
            self.weights = weights
        else:
            raise ValueError
    
    def fit(self,data,dataset_type='customer',scoring_method="Quintile",**kwargs):
        '''
        for model fitting: scoring_method - "Mean", "Median", or "Quintile"; scoring_method is set to "Quintile" by default.
        
        Accpeted **kwargs:
            "recency_scoring_method","frequency_scoring_method", and "monetary_scoring_method";
            Set the value of these **kwargs to the following defined scoring methods listed above.
        '''
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
        
        boo_0 = ((dataset_type=='customer')or(dataset_type =='transactional'))
        boo_1 = ((scoring_method == 'Mean') or (scoring_method == 'Median') or (scoring_method == 'Quintile'))
        # Here we do some initial parameter checking.
        if boo_0 and boo_1:
            #Instantiating the variables
            self.scoring_method = scoring_method
            self.dataset_type = dataset_type
            allowed_kwargs = ['recency_scoring_method','frequency_scoring_method','monetary_scoring_method']
            
            def catch_exception(values):
            	val = values[0]
            	if val in allowed_kwargs:
            		return values
            	else:
            		raise ValueError

            self.__dict__.update(catch_exception((k, v)) for k, v in kwargs.items())
            
            self.data = deepcopy(data)
            if boo_0:
                if data.shape[1] != 3:
                    print('Error: expected 3 columns for dataset type of '+ self.dataset_type +'; received: ' + str(data.shape[1]))
                    raise ValueError
            if self.dataset_type == 'transactional':
                self.data = convert_transaction_to_user(self.data)
            
            def get_func(func):
                func = func.lower()
                if func == 'quintile':
                    func = 'quantile'
                func = getattr(pd.Series,func)
                return func
            
            def score_variables(dataset,variable,class_dict,func):
                    dataset_copy = dataset
                    if func == pd.Series.quantile:
                        qnt = func(dataset_copy[variable],[0.2,0.4,0.6,0.8])
                        #print(qnt)
                        if 'recency' in variable.lower():
                            for i in np.arange(5,1,-1):
                                val0 = qnt[qnt.index[i-2]]
                                class_dict[i] = list(dataset_copy[dataset_copy[variable] >= val0].index)
                                dataset_copy = dataset_copy[dataset_copy[variable] < val0]
                            class_dict[i-1] = list(dataset_copy[variable].index)
                        else:
                            for i in range(0,4):
                                val0 = qnt[qnt.index[i]]
                                class_dict[i+1] = list(dataset_copy[dataset_copy[variable] <= val0].index)
                                dataset_copy = dataset_copy[dataset_copy[variable] > val0]
                            class_dict[i+2] = list(dataset_copy[variable].index)
                    #### Other case where the function of choice is not quintile.
                    else:
                        if 'recency' not in variable.lower():
                            for i in range(0,4):
                                divider = func(dataset_copy[variable])
                                class_dict[i+1] = list(dataset_copy[dataset_copy[variable] < divider].index)
                                dataset_copy = dataset_copy[dataset_copy[variable] >= divider]
                            class_dict[i+2] = list(dataset_copy[variable].index)
                        else:
                            for i in range(0,4):
                                divider = func(dataset_copy[variable])
                                class_dict[i+1] = list(dataset_copy[dataset_copy[variable] > divider].index)
                                dataset_copy = dataset_copy[dataset_copy[variable] <= divider]
                            class_dict[i+2] = list(dataset_copy[variable].index)
                    return class_dict
            
            def input_scores(dataset,dictionary,label):
                for key,index_list in dictionary.items():
                    dataset.loc[index_list,label] = key
                return dataset
            
            def apply_weights(dataset,weights):
                for col,w_label,weight in zip(dataset[labels],weighted_labels,weights):
                    dataset[w_label] = dataset[col] * weight
                return dataset
            
            #### Scoring Section
            recency_class_scores = {}
            frequency_class_scores = {}
            monetary_class_scores = {}
            class_dictionaries = [recency_class_scores,frequency_class_scores,monetary_class_scores]
            #Retrieving the indexes for each R-F-M categorized within each dictionary for each score
            data_ = deepcopy(self.data)
            kwargs_keys = ['recency_scoring_method','frequency_scoring_method','monetary_scoring_method']
            if len(self.__dict__) > 4:
                for col,dictionary,key in zip(data_,class_dictionaries,kwargs_keys):
                    try:
                        func = self.__dict__[key]
                        print("Scoring R-F-M using scoring method: '",func,"'...\n")
                        dictionary = score_variables(data_,col,dictionary,get_func(func))
                    except:
                        print("Scoring R-F-M using scoring method: '",self.scoring_method,"'...\n")
                        dictionary = score_variables(data_,col,dictionary,get_func(self.scoring_method))
                del(data_)
            else:
                print("Scoring R-F-M using scoring method: '",self.scoring_method,"'...\n")
                for col,dictionary in zip(data_,class_dictionaries):
                    dictionary = score_variables(data_,col,dictionary,get_func(self.scoring_method))
                del(data_)
            
            print('Placing Scores in dataset...\n')
            
            self.data['Recency_Scores'] = 0
            self.data['Frequency_Scores'] = 0
            self.data['Monetary_Scores'] = 0
            self.data['Recency_Weighted_Scores'] = 0
            self.data['Frequency_Weighted_Scores'] = 0
            self.data['Monetary_Weighted_Scores'] = 0
            labels = ['Recency_Scores','Frequency_Scores','Monetary_Scores']
            weighted_labels = ['Recency_Weighted_Scores','Frequency_Weighted_Scores','Monetary_Weighted_Scores']
            
            for label,dictionary in zip(labels,class_dictionaries):
                self.data = input_scores(self.data,dictionary,label)
            print('Applying weights to scores...\n')
            
            self.data = apply_weights(self.data,self.weights)
            print('Completed Fitting R-F-M.')
            #return self
        else:
            raise ValueError
                 
    
    def __str__(self):
        s = "Parameters:\n\tDataset_Type: "+str(self.dataset_type)+"\n\tScoring_Method: "+str(self.scoring_method)+"\n\tWeights: "+str(self.weights)+"\n"
        
        def print_kwargs(k_dict):
            string = ""
            if len(k_dict) > 4:
                string += "Kwargs:\n\t"
                for key,val in k_dict:
                    string += "\t"+str(key)+": "+str(val)+"\n"
            return string
        
        output = s + print_kwargs(self.__dict__)
        return output
    