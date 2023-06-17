# This utility file should alway go where the env file is.

# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Personal libraries
import acquire_
import env

#  Data preparation
# -----------------------------------------------------------------
def cleaning_telco() -> pd.DataFrame:
    """
    Goal: remove and clean columns
    perimenters:
        None
    return:
        telco: telco dataframe with removed duplicated columns and clean coulumn data types
    """

    # get my data
    telco, query = acquire_.get_telco_data()
    
    # lis of columns to remove
    drop_cols = ["customer_id", 
                 "internet_service_type_id", 
                 "contract_type_id",
                "payment_type_id"]

    # remove the columns
    telco = telco.drop(columns=drop_cols)
    
    # get the mean of all the rows of total charges that are actual numbers
    mean_of_digits_in_total_charges = telco[telco.total_charges.str.isdigit()].total_charges.astype("float").mean()


    # replace all the empty (" ") cells with the mean of the digit rows of total_charges
    # then convert the column into a float data type column
    telco["total_charges"] = telco["total_charges"].str.replace(" ", str(mean_of_digits_in_total_charges)).astype("float")

    return telco



def prep_telco_visuals(telco: pd.DataFrame) -> None:
    """
    Goal: return preparation visuals

    paremeters:
        telco: pandas data from that has been cleaned using the cleaning_telco function
    return:
        visuals of the numeric columns and object columns
    """
    # get all numeric column names
    numeric_cols = telco.select_dtypes("number").columns

    # create numeric columns value counts visuals
    for col in numeric_cols:

        # print value counts an normalized value counts
        print(col.upper())
        print(telco[col].value_counts(dropna=False))
        print(telco[col].value_counts(dropna=False, normalize=True))

        # show visuals of the value counts
        plt.figure(figsize=(5,3))
        sns.histplot(telco[col].value_counts(dropna=False))
        plt.show()

    # get all object column names
    object_cols = telco.select_dtypes("object").columns

    # create object columns value counts visuals
    for col in object_cols:

        # print value counts an normalized value counts
        print(col.upper())
        print(telco[col].value_counts(dropna=False))
        print(telco[col].value_counts(dropna=False, normalize=True))

        # show visuals of the value counts
        plt.figure(figsize=(5,3))
        sns.histplot(telco[col].value_counts(dropna=False))
        plt.show()


def final_prep_telco() -> pd.DataFrame:
    """
    Goal: generate fully clean data for exploration and modeling

    paremeters:
        None
    retunrn:
        telco: full prepared pandas datasete of telco_churn data
    """

    # get my data
    telco = cleaning_telco()

    # get all columns from dataframe
    all_columns = telco.columns

    # containers of different variable types
    categorical = []

    # separate variables
    for col in all_columns:
        # count number of unique valus in the column
        len_of_uniq = len(telco[col].unique())
        
        # also checking for only object data types
        if (col != "churn") and (len_of_uniq <= 5) and (telco[col].dtype == "O"):
            categorical.append(col)
        else: pass

    # generate dummy variables for each categorical column
    telco_dummies = pd.get_dummies(telco[categorical])

    # generate dummy variable for the target column
    target_dummy = pd.get_dummies(telco["churn"], drop_first=True)

    # get column name of all the dummies
    dummy_cols = telco_dummies.columns

    # clean the dummy columns by replacing "-" with "_" and make the name lower case
    dummy_cols = dummy_cols.str.replace("-", "_").str.lower()

    # concatinate the dummies to the telco data frame
    telco["churn"] = target_dummy
    telco[dummy_cols] = telco_dummies

    # add a base line column for modeling
    telco["baseline"] = int(telco.churn.mode())

    # remove all original cate gorical columns and re-assign telco
    telco = telco.drop(columns=categorical)

    return telco

    

# -----------------------------------------------------------------
# Split the data into train, validate and train
def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int =None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test