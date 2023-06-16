# This utility file should alway go where the env file is.

# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Personal libraries
import acquire_
import env

#  Data preparation
# -----------------------------------------------------------------
def prep_telco_data():
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