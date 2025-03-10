#!/usr/bin/env python
# coding: utf-8
# %%

# %%
get_ipython().run_line_magic('pip', 'install dowhy')


# %%


# %%


# imports
import networkx as nx 
import numpy as np
import pandas as pd
from dowhy import gcm

# loading the data 
depression = pd.read_csv("../../data/Student Depression Dataset.csv")
depression = depression.dropna()
depression = depression.replace({'Yes': 1, 'No': 0})

data_encoded = pd.get_dummies(depression, drop_first=True)
data_encoded = data_encoded[['Academic Pressure', 'Have you ever had suicidal thoughts ?',
'Financial Stress', 'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad',
'City_Hyderabad', 'City_Meerut', 'City_Patna', 'Dietary Habits_Moderate',
'Sleep Duration_7-8 hours', 'Sleep Duration_Less than 5 hours',
'Sleep Duration_More than 8 hours', 'Sleep Duration_Others',
'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Depression']]

depression_LASSO_features = data_encoded[['Academic Pressure', 'Have you ever had suicidal thoughts ?',
  'Financial Stress', 'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad',
  'City_Hyderabad', 'City_Meerut', 'City_Patna', 'Dietary Habits_Moderate',
  'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Depression']]

depression_LASSO_features = depression_LASSO_features.astype(int)



depression_LASSO_features['Dietary_Score'] = (
    depression_LASSO_features['Dietary Habits_Moderate'] * 1 +
    depression_LASSO_features['Dietary Habits_Others'] * 2 +
    depression_LASSO_features['Dietary Habits_Unhealthy'] * 0
)
X = depression_LASSO_features['Dietary_Score']

city_columns = ['City_Ahmedabad', 'City_Bhopal', 'City_Faridabad',
                'City_Hyderabad', 'City_Meerut', 'City_Patna']

depression_LASSO_features['City_Index'] = depression_LASSO_features[city_columns].idxmax(axis=1)
depression_LASSO_features['City_Index'] = depression_LASSO_features['City_Index'].apply(lambda x: 
                                                                                        city_columns.index(x) + 1)

Y = depression_LASSO_features['City_Index']
Z = depression_LASSO_features['Depression']



# this is what we are intervening on.
X = depression_LASSO_features['Academic Pressure']
Z = depression_LASSO_features['Depression']

# Construct the Graph, set up the model
causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([('X', 'Z')])) # X -> Z
causal_model.set_causal_mechanism('X', 
                                  gcm.EmpiricalDistribution())
causal_model.set_causal_mechanism('Z', 
                                  gcm.AdditiveNoiseModel(
                                      gcm.ml.create_linear_regressor()))

# Training data for the model,
# X -> Academic Pressure, Z -> Depression
training_data = pd.DataFrame(data=dict(X=X, Z=Z))

# fit the model to the training data
gcm.fit(causal_model, training_data)

gcm.counterfactual_samples( # generate counterfactual samples
    causal_model,
    {'X': lambda x: 5}, # intervene on Academic Pressure
    observed_data=pd.DataFrame(data=dict(X=[1], Z=[2])))


# %%




# %%


depression_LASSO_features.head()


# %%




# %%




# %%
def counterfactual_ap(int_num):

    # Construct the Graph, set up the model
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([
                                        ('Financial_Stress', 'Depression'),
                                        ('Suicidal_Thoughts', 'Depression'),
                                        ('Academic_Pressure', 'Depression')
                                        ])
                                        )

    causal_model.set_causal_mechanism('Financial_Stress', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Suicidal_Thoughts', 
                                      gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Academic_Pressure', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Depression', 
                                      gcm.AdditiveNoiseModel(
                                          gcm.ml.create_linear_regressor()))

    # Fix Financial Stress, Suicidal Thoughts
    training_data = pd.DataFrame(data=dict(
        **{'Financial_Stress': depression_LASSO_features['Financial Stress']},
        **{'Suicidal_Thoughts': depression_LASSO_features['Have you ever had suicidal thoughts ?']},
        **{'Academic_Pressure': depression_LASSO_features['Academic Pressure']},
        **{'Depression': depression_LASSO_features['Depression']}
        ))

    # print(training_data.head())
    # print("Columns in training_data:", training_data.columns)

    # fit the model to the training data
    gcm.fit(causal_model, training_data)

    # generate counterfactual samples 
    counterfactual_result = gcm.counterfactual_samples(
        causal_model,
        {'Academic_Pressure': lambda x: int_num}, # intervene on Academic Pressure, set to 4
        observed_data=pd.DataFrame(data=dict(
            Financial_Stress=[1], # fix Financial Stress
            Suicidal_Thoughts=[1], # fix Suicidal Thoughts
            Academic_Pressure=[5],
            Depression=[1]))) 

    return counterfactual_result


# %%
def counterfactual_fs(int_num):

    # Construct the Graph, set up the model
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([
                                        ('Financial_Stress', 'Depression'),
                                        ('Suicidal_Thoughts', 'Depression'),
                                        ('Academic_Pressure', 'Depression')
                                        ])
                                        )

    causal_model.set_causal_mechanism('Financial_Stress', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Suicidal_Thoughts', 
                                      gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Academic_Pressure', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Depression', 
                                      gcm.AdditiveNoiseModel(
                                          gcm.ml.create_linear_regressor()))

    # Fix Academic Pressure, Suicidal Thoughts
    training_data = pd.DataFrame(data=dict(
        **{'Financial_Stress': depression_LASSO_features['Financial Stress']},
        **{'Suicidal_Thoughts': depression_LASSO_features['Have you ever had suicidal thoughts ?']},
        **{'Academic_Pressure': depression_LASSO_features['Academic Pressure']},
        **{'Depression': depression_LASSO_features['Depression']}
        ))

    # print(training_data.head())
    # print("Columns in training_data:", training_data.columns)

    # fit the model to the training data
    gcm.fit(causal_model, training_data)

    # generate counterfactual samples 
    counterfactual_result = gcm.counterfactual_samples(
        causal_model,
        {'Financial_Stress': lambda x: int_num}, # intervene on Financial Stress, set to 4
        observed_data=pd.DataFrame(data=dict(
            Financial_Stress=[1],
            Suicidal_Thoughts=[1], # fix Suicidal Thoughts
            Academic_Pressure=[5], # fix Academic Pressure
            Depression=[1]))) 

    return counterfactual_result


# %%

def counterfactual_st(int_num):
    # Construct the Graph, set up the model
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([
                                        ('Financial_Stress', 'Depression'),
                                        ('Suicidal_Thoughts', 'Depression'),
                                        ('Academic_Pressure', 'Depression')
                                        ])
                                        )

    causal_model.set_causal_mechanism('Financial_Stress', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Suicidal_Thoughts', 
                                      gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Academic_Pressure', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Depression', 
                                      gcm.AdditiveNoiseModel(
                                          gcm.ml.create_linear_regressor()))

    # Fix Academic Pressure, Suicidal Thoughts
    training_data = pd.DataFrame(data=dict(
        **{'Financial_Stress': depression_LASSO_features['Financial Stress']},
        **{'Suicidal_Thoughts': depression_LASSO_features['Have you ever had suicidal thoughts ?']},
        **{'Academic_Pressure': depression_LASSO_features['Academic Pressure']},
        **{'Depression': depression_LASSO_features['Depression']}
        ))

    # print(training_data.head())
    # print("Columns in training_data:", training_data.columns)

    # fit the model to the training data
    gcm.fit(causal_model, training_data)

    # generate counterfactual samples 
    counterfactual_result = gcm.counterfactual_samples(
        causal_model,
        {'Suicidal_Thoughts': lambda x: 0}, # intervene on Suicidal Thoughts, set to 0
        observed_data=pd.DataFrame(data=dict(
            Financial_Stress=[1], # fix Financial Stress
            Suicidal_Thoughts=[1], 
            Academic_Pressure=[5], # fix Academic Pressure 
            Depression=[1]))) 

    return counterfactual_result


# - Suicidal Thoughts intervention -> set to 0, 
# - fix rest: academic pressure = 5, financial stress = 1
# - result: depression decreased to 0.579094

# > intervening on a variable that did not make the cut from LASSO;
# > Sleep Duration.
# - column names:  'Sleep Duration_7-8 hours',
#  'Sleep Duration_Less than 5 hours',
#  'Sleep Duration_More than 8 hours',
#  'Sleep Duration_Others'

# %%
#for which_sleep, input 1 to test less than 5 hours, 2 for 7-8, 3 for more than 8, 4 for other
def counterfactual_sd(int_num, which_sleep):

    # Construct the Graph, set up the model
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([
                                        ('Financial_Stress', 'Depression'),
                                        ('Suicidal_Thoughts', 'Depression'),
                                        ('Academic_Pressure', 'Depression'),
                                        ('Sleep Duration_7-8 hours', 'Depression'),
                                        ('Sleep Duration_Less than 5 hours', 'Depression'),
                                        ('Sleep Duration_More than 8 hours', 'Depression'),
                                        ('Sleep Duration_Others', 'Depression')
                                        ])
                                        )

    causal_model.set_causal_mechanism('Financial_Stress', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Suicidal_Thoughts', 
                                      gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Academic_Pressure', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Sleep Duration_7-8 hours', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Sleep Duration_Less than 5 hours', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Sleep Duration_More than 8 hours', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Sleep Duration_Others', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Depression', 
                                      gcm.AdditiveNoiseModel(
                                          gcm.ml.create_linear_regressor()))

    # Fix Academic Pressure, Suicidal Thoughts, Financial Stress
    training_data = pd.DataFrame(data=dict(
        **{'Financial_Stress': data_encoded['Financial Stress']},
        **{'Suicidal_Thoughts': data_encoded['Have you ever had suicidal thoughts ?']},
        **{'Academic_Pressure': data_encoded['Academic Pressure']},
        **{'Sleep Duration_7-8 hours': data_encoded['Sleep Duration_7-8 hours']},
        **{'Sleep Duration_Less than 5 hours': data_encoded['Sleep Duration_Less than 5 hours']},
        **{'Sleep Duration_More than 8 hours': data_encoded['Sleep Duration_More than 8 hours']},
        **{'Sleep Duration_Others': data_encoded['Sleep Duration_Others']},
        **{'Depression': data_encoded['Depression']}
        ))

    # fit the model to the training data
    gcm.fit(causal_model, training_data)
    
    int_1 = 0
    int_2 = 0
    int_3 = 0
    int_4 = 0
    
    if which_sleep == 1:
        int_1 = int_num
    elif which_sleep == 2:
        int_2 = int_num
    
    elif which_sleep == 3:
        int_3 = int_num
    else:
        int_4 = int_num
        

    # generate counterfactual samples 
    counterfactual_result = gcm.counterfactual_samples(
        causal_model, 
        {'Sleep Duration_7-8 hours': lambda x: int_2,
        'Sleep Duration_Less than 5 hours': lambda x: int_1 ,
        'Sleep Duration_More than 8 hours': lambda x: int_3,
        'Sleep Duration_Others': lambda x: int_4}, # intervene on sleep duration
        observed_data=pd.DataFrame(data={
            'Financial_Stress': [1], # fix Financial Stress
            'Suicidal_Thoughts': [1], # fix Suicidal Thoughts
            'Academic_Pressure': [5], # fix Academic Pressure
            'Sleep Duration_7-8 hours': [0],  
            'Sleep Duration_Less than 5 hours': [1],  # Assume original state was "<5 hours"
            'Sleep Duration_More than 8 hours': [0],  
            'Sleep Duration_Others': [0],   
            'Depression': [1]
            }
            ))

    return counterfactual_result


# > intervening on Sleep Duration:
# - original observed value: Sleep Duration < 5 hours
# - intervention: Sleep Duration 7-8 hours; fixed Academic Pressure at 5, FS at 1, Suicidal Thoughts at 1
#     - result: Depression = 0.966465. Note: fixing FS at 5 did not change anything either.
