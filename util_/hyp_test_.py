from math import exp
from termios import PARODD
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# system import to access file on different directory 
import sys
sys.path.append("/Users/migashane/CodeUp/Data_Science/telco_churn_classification_project/util_")
import prepare_
import explore_

# set a default them for all my visuals
sns.set_theme(style="whitegrid")

# load data
telco = prepare_.clean_telco_without_dummies()

# split data into train, validate and test
train, validate, test = prepare_.split_data_(df=telco,
                     test_size=0.2, 
                     validate_size=0.2,
                    stratify_col="churn",
                    random_state=95)

def internet_service_with_churn(train = train) -> pd.DataFrame:
    """
    Goal: to retrieve all information about my stats test on internet service typs
    """

    # print("Null_hyp: The type of internet service does not significantly impact the likelihood of churn?")
    # print("Alt_hyp: The type of internet service significantly impact the likelihood of churn?")

    # run a contegency tale
    cont_table = pd.crosstab(train.churn, train.internet_service_type)
    
    # set significance level
    alpha = 0.05

    # test stats
    chi2, p_value, degreeFreedom, exp_table = stats.chi2_contingency(cont_table)

    # print results
    # print("chi2:", chi2)
    # print("p-value:", p_value)
    # print("defrees of freedom:", degreeFreedom, "\n\n")

    # print(cont_table)
    # # oompare p-value to alpha
    # if p_value < alpha:
    #     print("We have enough evidence to reject the null")
    # else:
    #     print("we fail to reject the null at this time")
    expected = pd.DataFrame(exp_table).astype(int)
    return cont_table,  expected


def monthly_charges_and_tenure(train: pd.DataFrame=train):
    """
    Goal: To answer the following question with hypotheis testing...
        What month are customers most likely to churn and does that depend on their contract type?.¶
    
    Null_hyp: The churn month is independent (no association) of the contract type.
    Alt_hyp The churn month is dependent (association) of the contract type.
    """
    
    # retrieve my two groups
    first_24 = train[train.tenure <= 24].tenure
    contract_type = train[train.tenure <= 24].contract_type

    # generate contingency table
    cont_table = pd.crosstab(contract_type, first_24)

    # set significance level
    alpha = 0.05

    # test stats
    chi2, p_value, degreeFreedom, exp_table = stats.chi2_contingency(cont_table)

    # print results
    # print("chi2:", chi2)
    # print("p-value:", p_value)
    # print("defrees of freedom:", degreeFreedom, "\n\n")

    # print("Expected")
    # pd.DataFrame(exp_table)

    return round(cont_table - exp_table, 0)

def partner_or_dependents(train:pd.DataFrame= train):
    """
    Goal: To answer the following question with hypotheis testing...
        Does the customer having a partner or dependents affect churn.¶

    Null_hyp: A customer having a parner does not affect churn.
    Alt_hyp: A customer having aparter does affect churn.
    """

    # run a contegency tale
    cont_table = pd.crosstab(train.churn, train.partner)

        # set significance level
    alpha = 0.05

    # test stats
    chi2, p_value, degreeFreedom, exp_table = stats.chi2_contingency(cont_table)

    # print results
    # print("chi2:", chi2)
    # print("p-value:", p_value)
    # print("defrees of freedom:", degreeFreedom, "\n\n")

    # print(cont_table)
    # pd.DataFrame(exp_table)

    return round(cont_table - exp_table, 0)


def tech_support_vs_churn(train=train):
    """
    Goal: to retrieve all information about my stats test on technical suport vs churn
    """

    # print("Null_hyp: Customers with technical support have the same or higher churn rate as those without it.")
    # print("Alt_hyp: Customers with technical support have lower churn rate compared to those without it.")

    # run a contegency tale
    cont_table = pd.crosstab(train.churn, train.tech_support)
    
    # set significance level
    alpha = 0.05

    # test stats
    chi2, p_value, degreeFreedom, exp_table = stats.chi2_contingency(cont_table)

    # # print results
    # print("chi2:", chi2)
    # print("p-value:", p_value)
    # print("defrees of freedom:", degreeFreedom, "\n\n")

    # print(cont_table)
    exp_table = pd.DataFrame(exp_table).astype(int)
    # exp_table
    return cont_table, exp_table