# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:25:52 2018

@author: juan.zarco
"""
from copy import deepcopy
import sys
try:
    from rfm_utils import *
except ImportError:
    from RFM.rfm_utils import *

accepted_keys = ['recency_scoring_method', 'frequency_scoring_method', 'monetary_scoring_method']
accepted_values = ['mean', 'median', 'quintile']

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
        if sum(weights) == 1 and len(weights) == 3:
            self.weights = weights
        else:
            raise ValueError
    
    def fit(self,data,dataset_type='customer',scoring_method="quintile",recency_end_date=None,**kwargs):
        '''
        for model fitting: scoring_method - "mean", "median", or "quintile"; scoring_method is set to "quintile" by default.

        Recency_end_date: Accepted date formats - "01/01/2010" or "01-01-2010" i.e. "Month - Day - Year"; This is to be used if dataset_type is transactional otherwise it is ignored.
        
        Accpeted **kwargs:
            "recency_scoring_method","frequency_scoring_method", and "monetary_scoring_method";
            Set the value of these **kwargs to the following defined scoring methods listed above.

        dataset_type: "customer" dataset should be labeled row by row for each individual user or consumer. Three columns should be listed
            containing: Number of days since last transaction, Number of Transactions occurred, and the total monetary value
            of the transactions. Each column should be labeled 'recency', 'frequency', and 'monetary'

            "transactional" dataset should be a log of all transactions labeled by a User Identification (first column). Two columns should be
            listed containing: date of each transactions (second column) and the monetary value of each transaction (third column).
        '''

        ####CHECK PARAMETERS ARE VALID####

        self.scoring_method = scoring_method.lower()
        self.dataset_type = dataset_type.lower()
        if recency_end_date is not None:
            try:
                self.recency_end_date = datetime.strptime(recency_end_date, "%m/%d/%Y")
            except:
                try:
                    self.recency_end_date = datetime.strptime(recency_end_date, "%m-%d-%Y")
                except:
                    try:
                        self.recency_end_date = datetime.strptime(recency_end_date, "%d-%b-%y")
                    except Exception:
                        raise Exception("Unexpected error: {}".format(sys.exc_info()[0]))

        if not ((dataset_type=='customer')or(dataset_type =='transactional')):
            raise ValueError('Unexpected value for dataset_type: {}'.format(dataset_type))
        elif not ((self.scoring_method == 'mean') or (self.scoring_method == 'median') or (self.scoring_method == 'quintile')):
            raise ValueError('Unexpected value for scoring_method')



        for key, val in kwargs.items():
            if key not in accepted_keys:
                raise ValueError('Unexpected key inputted: {}'.format(key))
            elif val not in accepted_values:
                raise ValueError('Unexpected value inputted: {}'.format(val))
            else:
                self.__dict__.update({key: val})

        self.data = deepcopy(data)
        if self.dataset_type == 'customer':
            if self.data.shape[1] != 3:
                raise ValueError('Error: expected 3 columns for dataset type of '+ self.dataset_type +'; received: ' + str(data.shape[1]))

        if self.dataset_type == 'transactional':
            self.data = convert_transaction_to_user(self.data, recency_end_date=recency_end_date)

        #### Scoring Section
        #Retrieving the indexes for each R-F-M categorized within each dictionary for each score

        self.cutoffs = {}
        if 'recency_scoring_method' in self.__dict__:
            recency_cutoffs = build_cutoffs(self.data, 'recency', self.__dict__['recency_scoring_method'])
        else:
            recency_cutoffs = build_cutoffs(self.data, 'recency', self.scoring_method)

        if 'frequency_scoring_method' in self.__dict__:
            frequency_cutoffs = build_cutoffs(self.data, 'frequency', self.__dict__['frequency_scoring_method'])
        else:
            frequency_cutoffs = build_cutoffs(self.data, 'frequency', self.scoring_method)

        if 'monetary_scoring_method' in self.__dict__:
            monetary_cutoffs = build_cutoffs(self.data, 'monetary', self.__dict__['monetary_scoring_method'])
        else:
            monetary_cutoffs = build_cutoffs(self.data, 'monetary', self.scoring_method)
        self.cutoffs.update({'recency': recency_cutoffs})
        self.cutoffs.update({'frequency': frequency_cutoffs})
        self.cutoffs.update({'monetary' : monetary_cutoffs})

        self.data = input_scores(self.data, recency_cutoffs, 'recency')
        self.data = input_scores(self.data, frequency_cutoffs, 'frequency')
        self.data = input_scores(self.data, monetary_cutoffs, 'monetary')

        self.data = apply_weights(self.data, ['recency', 'frequency', 'monetary'], self.weights)

    def get_fitted_data(self):
        return self.data

    def summary_statistics(self):
        n_obs = self.data.shape[0]
        recency_summary = self.data['recency_scores'].value_counts() / n_obs
        frequency_summary = self.data['frequency_scores'].value_counts() / n_obs
        monetary_summary = self.data['monetary_scores'].value_counts() / n_obs

        summaries = [recency_summary, frequency_summary, monetary_summary]

        summary_df = pd.concat(summaries, axis=1)

        return summary_df.fillna(0)

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
    