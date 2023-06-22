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



def telco_multivariate_visuals():
    """
    Goal: retrieve all the mulivariate visuals for the telco training dataset
    """
    # get a pair plot of all my numeric variable
    sns.pairplot(train)

    # gather all my numeric columns data
    numerics_data = train.select_dtypes("number")

    # Silence Seaborn warnings
    warnings.filterwarnings("ignore", module="seaborn")

    # get the combinations of numeric columns in pairs
    combinations = []
    combinations.extend(itertools.combinations(numerics_data.columns, 2))

    #Note: here I am still using the training data not the melted
    target = "churn"
    for cols in combinations:
        plt.figure(figsize=(5,3))
        
        # plot continuous vs continuous
        if len(train[cols[0]].value_counts()) > 5 and len(train[cols[1]].value_counts()) > 5:
            print(cols[0].upper(), "vs", cols[1].upper())
            
            sns.relplot(x=train[cols[0]], y=train[cols[1]], col=train[target])
            plt.show()
        else:
            print(cols[0].upper(), "vs", cols[1].upper())

            sns.boxenplot(x=train[cols[0]], y=train[cols[1]], hue=train[target])
            plt.show()

    # gather all my numeric columns data
    columns = train.select_dtypes("number").columns

    plt.figure(figsize=(5,3))
    # First plot with all categories
    # melt the data from a wide format to long
    melted_numerics = train[columns].melt(id_vars="churn")
    sns.boxplot(data= melted_numerics, x="variable", y="value", hue="churn")
    plt.title("All numerics vs churn")
    plt.show()


    # remove total_charges from the data
    numerics_no_tCharges = []
    for i in columns:
        if i == "total_charges":
            pass
        else:
            numerics_no_tCharges.append(i)

    # all but tatal charges
    # melt the data from a wide format to long
    plt.figure(figsize=(5,3))
    melted_numerics = train[numerics_no_tCharges].melt(id_vars="churn")
    sns.boxplot(data= melted_numerics, x="variable", y="value", hue="churn")
    plt.title("No total charges vs churn")
    plt.show()



    # remove monthly charges from the data
    numerics_no_mCharges = []
    for i in columns:
        if i in ["monthly_charges","total_charges"]:
            pass
        else:
            numerics_no_mCharges.append(i)

    # all but tatal charges
    # melt the data from a wide format to long
    plt.figure(figsize=(5,3))
    melted_numerics = train[numerics_no_mCharges].melt(id_vars="churn")
    sns.boxplot(data= melted_numerics, x="variable", y="value", hue="churn")
    plt.title("No monthly charges vs churn")
    plt.show()


def _verify_alpha_(p_value, alpha=0.05) -> None:
    """
    Goal: Test if we are acepting or rejecting the null
    """
    # oompare p-value to alpha
    if p_value < alpha:
        print("We have enough evidence to reject the null")
    else:
        print("we fail to reject the null at this time")


def churn_depend_on_contract():
    """
    Goal: to retrieve all information about my stats test on churn month with contract type
    """
    # load data
    telco = prepare_.clean_telco_without_dummies()

    # split data into train, validate and test
    train, validate, test = prepare_.split_data_(df=telco,
                        test_size=0.2, 
                        validate_size=0.2,
                        stratify_col="churn",
                        random_state=95)


    print("Question: 2.1 First, is there a linear relationship between monthly charges and tenure.")
    print("Null_hyp: There is not linear relationship between monthly charges and tenure.")
    print("Alt_hyp: There is linear relationship between monthly charges and tenure..")

    # plot the columns
    sns.relplot(data=train, x="monthly_charges", y="tenure", hue="churn",col="churn")

    # confidence level
    alpha = 0.05

    # perform the pearson's rcorrelation test
    r, p_value = stats.pearsonr(train.monthly_charges, train.tenure)

    # print results
    print("coeficient r:", r)
    print("p-value:", p_value)
    _verify_alpha_(p_value) # compare p-value to alpha


    print("Question: 2.2 Are customers more or less likely to churn within the first 24 month with Telc")
    print("Null_hyp: Customer churn rate with in the first 24 month is > 50%.")
    print("Alt_hyp: Customer churn rate with in the first 24 month is <= 50%")

    # separate firt 24 month from all other
    first_24 = train[train.tenure <= 24]
    other = train[train.tenure > 24]

    # create mean and stanrd deviation for the two groups
    first_24_mean, first_24_std = first_24.tenure.mean(), first_24.tenure.std()
    other_mean, other_std = other.tenure.mean(), other.tenure.std()

    # create distribution objects
    first_24_dist = stats.norm(first_24_mean, first_24_std)
    other_dist = stats.norm(other_mean, other_std)

    # generate random variables
    first_24_rand = first_24_dist.rvs(10_000)
    other_rand = other_dist.rvs(10_000)

    #plot
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    sns.histplot(first_24_rand, ax=ax[0])
    sns.histplot(other_rand, ax=ax[1])
    plt.show()

    # confidence level
    alpha = 0.05

    # test statistics
    t_stats, p_value = stats.ttest_1samp(first_24_mean, other_mean)

    # print results
    print("test stats:", t_stats)
    print("p-Value:",  p_value)

    # oompare p-value to alpha
    if (t_stats > 0) and (p_value < alpha):
        print("We have enough evidence to reject the null")
    else:
        print("we fail to reject the null at this time")


    print("Main Question: What month are customers most likely to churn and does that depend on their contract type?")
    print("Null_hyp:The churn month is independent (no association) of the contract type.")
    print("Alt_hyp: The churn month is dependent (association) of the contract type")


        # retrieve my two groups
    first_24 = train[train.tenure <= 24].tenure
    contract_type = train[train.tenure <= 24].contract_type

    cont_table = pd.crosstab(contract_type, first_24)

    # set significance level
    alpha = 0.05

    # test stats
    chi2, p_value, degreeFreedom, exp_table = stats.chi2_contingency(cont_table)

    # print results
    print("chi2:", chi2)
    print("p-value:", p_value)
    print("defrees of freedom:", degreeFreedom, "\n\n")

    # conclusion
    _verify_alpha_(p_value)

    return pd.DataFrame(exp_table)