# This utility file should alway go where the env file is.

# For funtion annotations
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mac libraries
import os

# Personal libraries
import env

# Data acquisition
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# get data from codeup data sql database
def get_codeup_sql_data_(db_name: str, table_name: str = None, query: str= None) -> Tuple[pd.DataFrame, str]:
    """
    paremeters:
        db_name: name of the database you wich to access
        table_name: name of table you are quering from
        query: (optional argument) the query you want to retrieve
        
        note: enter query or table name NOT BOTH

    return:
        data: panads data frame fromt sql query
        query: the query used to retreive the data
    """
    if table_name: # if table is given
        query=f"""
            SELECT *
            FROM {table_name};
            """
        # access the database and retreive the data
        data = pd.read_sql(query, env.get_db_access(db_name))
    elif query:
        # access the database and retreive the data
        data = pd.read_sql(query, env.get_db_access(db_name))

    return data, query # return both the data and the query


def get_telco_data():
    """
    Goal: To return my acquired data from the codeup database and make it available to preparetion
    """

    query_ = """
        SELECT *
        FROM customers #payment_types
        JOIN contract_types ct USING(contract_type_id)
        JOIN internet_service_types ist USING(internet_service_type_id)
        JOIN payment_types pt USING(payment_type_id);
        """

    # get telco data from codeup database
    telco, query = get_codeup_sql_data_(db_name="telco_churn", query=query_)
    return telco, query