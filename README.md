# Modeling-Modules

Following Repository contains packages for different models for different purposes listed below written in Python.

This package should be used with the intent of a CRM analysis in order to score their CRM dataset with
the well known method of RFM.

RFM (Recency - Frequency - Monetary):
  - RFM Package used to model and score CRM data with the following scoring methods: Quintile Scoring, Mean Scoring, and Median Scoring.
    The package allows custom scoring on each variable related to Recency, Frequency, and Monetary. It is important to format the data 
    when fitting the model depending on the dataset type listed below:
    
    Customer/User Level CRM Dataset:
      - The dataset should contain the Customer IDs to score against as an index. The Recency, Frequency, and Monetary columns should be
        labeled as so in order to score the variables and customers correctly.
        
    Transactional CRM Dataset:
      - The dataset should have a list of Cutomer IDs associated with each transaction as well as a date that can be interpreted by Pythons
        datetime package and a column associated the value of the transaction. The columns should be in order as so. There is no need to set
        the Customer IDs as the index as it will be set automatically during the scoring.
