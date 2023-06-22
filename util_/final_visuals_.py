import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# system import to access file on different directory 
import sys
sys.path.append("/Users/migashane/CodeUp/Data_Science/telco_churn_classification_project/util_")

# wrangle and eda files
import acquire_
import prepare_
import explore_
import hyp_test_

# acquiring, cleaning, and adding features to data
telco = prepare_.final_prep_telco()

# splitting data into train, validate, and test
train, validate, test = prepare_.split_data_(df=telco,
                     test_size=0.2, 
                     validate_size=0.2,
                    stratify_col="churn",
                    random_state=95)

# look at the different splits
print("Shape of splits:",train.shape, validate.shape, test.shape)


def internet_service_impact_on_churn_visual(train=train):
    """
    Goal: retreive the final visual for the following question...
         Does the type of internet service significantly impact the likelihood of churn?
    """

    actual, expected = hyp_test_.internet_service_with_churn()

    # Plot the reuslt of internet service type with churn
    fig, ax = plt.subplots(1,2,figsize=(10,4))

    # first plot
    act_cmap = sns.color_palette("Blues") # color selection
    act = sns.heatmap(actual, annot=True, fmt='d', cmap=act_cmap, ax=ax[0])
    act.set_title("Ectual")

    # second plot
    exp_cmap = sns.color_palette("Greens") # color selectiono
    exp = sns.heatmap(expected, annot=True, fmt='d', cmap=exp_cmap, ax=ax[1])
    exp.set_title("Expected")

    # add super title
    fig.suptitle('Fiber optic customers are churning?')
    plt.show()


def tech_support_vs_churn_visual(train=train):
    """
    Goal: retreive the final visual for the following question...
         Do customers with technical support have a lower churn rate compared to those without it?¶
    """

    actual, expected = hyp_test_.tech_support_vs_churn()

    # Plot the reuslt of internet service type with churn
    fig, ax = plt.subplots(1,2,figsize=(10,4))

    # first plot
    act_cmap = sns.color_palette("Blues") # color selection
    act = sns.heatmap(actual, annot=True, fmt='d', cmap=act_cmap, ax=ax[0])
    act.set_title("Ectual")

    # second plot
    exp_cmap = sns.color_palette("Greens") # color selectiono
    exp = sns.heatmap(expected, annot=True, fmt='d', cmap=exp_cmap, ax=ax[1])
    exp.set_title("Expected")

    # add super title
    fig.suptitle("Customers who don't receive technical suport are leaving?")
    plt.show()

def role_of_contract_type_visual(train=train):
    """
    Goal: retreive the final visual for the following question...
        What role does contract type play in the first 24 month?
    """
    # separate my frist 24 month tenure from all the other months.
    first_24 = train[(train.tenure <= 24) & (train.churn == 1)]

    # group count of churn custmes by contract type
    churn_first_24_by_contract = first_24.groupby("contract_type").churn.agg("sum")

    # convert to pandas data frame
    churn_first_24_by_contract = pd.DataFrame(churn_first_24_by_contract)

    # reset the index of the churn_first_24_by_contract dataframe
    melted_table = churn_first_24_by_contract.reset_index()

    # plot the result
    plt.figure(figsize=(5,4))
    sns.barplot(data = melted_table, x= "contract_type", y="churn")
    plt.title("Month-to-month customers churn in the first 2-years!")
    plt.show()

def monthly_charges_and_tenure_vusual(train=train):
    """
    Goal: retreive the final visual for the following question...
        Is there a linear relationship between monthly charges and tenure?¶
    """

    # separate my frist 24 month tenure from all the other months.
    churn = train[(train.churn == 1)]

    # Plot the result
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=churn, x="tenure", y="monthly_charges")
    plt.title("Monthly charges and tenure are affecting churn?")
    plt.show()

def churn_in_first_24_month_visual(train=train):
    """
    Goal: retreive the final visual for the following question...
        Are customers more or less likely to churn within the first 24 month with Telco?
    """
    # separate my frist 24 month tenure from all the other months.
    after_24 = train[(train.tenure > 24) & (train.churn == 1)]
    first_24 = train[(train.tenure <= 24) & (train.churn == 1)]

    # plot the result
    fig, ax = plt.subplots(1,2, figsize= (8,4))

    # first plot 
    first = sns.scatterplot(data = first_24, x = "tenure", y = "monthly_charges", ax=ax[0])
    first.set_title("first 24 month")

    # second plot
    last = sns.scatterplot(data = after_24, x = "tenure", y = "monthly_charges", ax=ax[1])
    last.set_title("After first 24 month")

    # add title
    fig.suptitle('Customers are churning within the first 24 month?')
    plt.show()
