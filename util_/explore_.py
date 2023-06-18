import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# system import to access file on different directory 
import sys
sys.path.append("/Users/migashane/CodeUp/Data_Science/telco_churn_classification_project/util_")
import prepare_

# set a default them for all my visuals
sns.set_theme(style="whitegrid")



def telco_univariate_visuals() -> None:
    """
    Goal: generate univariate statistics visuals for the telco trainingfeatures
    """
    # Get telco data
    telco = prepare_.clean_telco_without_dummies()

    # split data into train, validate and test
    train, validate, test = prepare_.split_data_(df=telco,
                        test_size=0.2, 
                        validate_size=0.2,
                        stratify_col="churn",
                        random_state=95)
    
    # separeate discrete from continuous variables
    continuous_col = []
    categorical_col = []

    for col in train.columns:
        if train[col].dtype == "O":
            categorical_col.append(col)

        else:
            if len(train[col].unique()) < 5: #making anything with less than 4 unique values a catergorical value
                categorical_col.append(col)
            else:
                continuous_col.append(col)

    # create visuals for each continuous varable
    for ele in continuous_col:
        print(ele.upper())
        print(train[ele].describe())
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        
        # first plot
        sns.histplot(train[ele], ax=ax[0])
        
        # second plot
        sns.kdeplot(train[ele], ax=ax[1])
        
        plt.tight_layout()
        plt.show()

    # create visuals for each continuous varable
    for ele in categorical_col:
        print(ele.upper())
        print(train[ele].describe())
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        
        # first plot
        sns.countplot(data=train, x = ele, ax=ax[0])
        
        # secondplot
        sns.boxplot(train[ele].value_counts(), ax=ax[1])
        
        plt.tight_layout()
        plt.show()


def telco_bivariate_stats_visuals() -> None:
    """
    Goal: retrieve all the bivariate visuals for the telco training dataset
    """
    # get combination of all columns paired with the target column
    columns = train.columns
    target = "churn"
    combinations = []
    for element in columns:
        if element == "churn":
            pass
        else:
            combinations.append((target, element))


    # create a dummy for churn
    train["churn"]= pd.get_dummies(train.churn, drop_first=True)

    # generate visuals for bivariate statistics
    for combo in combinations:
        # descriptive statistics
        print(combo[0].upper(), "vs", combo[1].upper())
        print(train[combo[1]].describe())
        
        # create a subplot object
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        
        # first visual
        sns.barplot(data=train, x=combo[0], y=combo[1], ax=ax[0,0])
        
        # second visual
        sns.stripplot(data=train, x=combo[0], y=combo[1], ax=ax[0,1])
        
        # third visual
        sns.boxplot(data=train, x=combo[0], y=combo[1], ax=ax[1,0])
        
        # fourth visual
        sns.violinplot(data=train, x=combo[0], y=combo[1], ax=ax[1,1])
        
        plt.tight_layout()
        plt.show()



