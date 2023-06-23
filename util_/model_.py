import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# imports for modeling
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# for knn
from sklearn.neighbors import KNeighborsClassifier

# for decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# for random forest
from sklearn.ensemble import RandomForestClassifier

# for logistic regression
from sklearn.linear_model import LogisticRegression

# system import to access file on different directory 
import sys
sys.path.append("/Users/migashane/CodeUp/Data_Science/telco_churn_classification_project/util_")
import prepare_
import knn_model_


# get the fully cleaned data with dummies
telco = prepare_.final_prep_telco()

# split data into train, validate and test
train, validate, test = prepare_.split_data_(df=telco,
                     test_size=0.2, 
                     validate_size=0.2,
                    stratify_col="churn",
                    random_state=95)

# features to pass to my model
features = ["internet_service_type_none", "internet_service_type_fiber optic", 
            "internet_service_type_dsl", "monthly_charges", "tenure", "contract_type_one year",
           "contract_type_two year", "contract_type_month_to_month", "phone_service_no", "phone_service_yes",
           "paperless_billing_no", "paperless_billing_yes", "tech_support_no", "tech_support_yes", "tech_support_no_internet_service"]

# the internet type feature dummy columns for modeling for training
xtrain = train[features]
ytrain = train.churn

# the internet type feature dummy columns for modeling for validation
xvalidate = validate[features]
yvalidate = validate.churn

# the internet type feature dummy columns for modeling for testing
xTest = test[features]
yTest = test.churn


def knn_model(xtrain = xtrain, ytrain = ytrain, xvalidate=xvalidate, yvalidate=yvalidate):
    """
    Goal: Train KNN model on different k values and return the best model
    """

    # the maximun number of neighbors the model should look at
    # in my case it can only look at 1% of the data
    k_neighbors = math.ceil(len(train) * 0.01)

    # the final result metric
    metrics = []

    for k in range(1, k_neighbors + 1):
        train_baseline_acc_score = accuracy_score(train.churn, train.baseline)
        
        # create a knn object
        #                          n_neighborsint(default=5) 
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)
        #                                                        p=1 uses the manhattan distance

        # fit training data to the object
        knn = knn.fit(xtrain, ytrain)
        
        #USE the thing
        train_score= knn.score(xtrain, ytrain)
        validate_score = knn.score(xvalidate, yvalidate)
        
        # create a dictionary of scores
        output = {
            "k": k,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": train_baseline_acc_score - train_score,
            "validate_baseline_diff": train_baseline_acc_score - validate_score,
            "baseline_accuracy": train_baseline_acc_score,
        }
        
        metrics.append(output)
    # get the result as a dataframe
    model_df = pd.DataFrame(metrics)

    # find the model with smallest ifference between the train and validate
    bestModel = model_df[model_df.difference == abs(model_df.difference).min()]
    
    # return the best knn model
    return bestModel


def decision_tree_model(xtrain = xtrain, ytrain = ytrain, xvalidate=xvalidate, yvalidate=yvalidate):
    """
    Goal: Train decision tree model on different tree depth and return the best model
    """

    metrics = []
    for d in range(1,11):
    #     base line
        train_baseline_acc_score = accuracy_score(train.churn, train.baseline)

    #      create tree object
        treeClf = DecisionTreeClassifier(max_depth= d, random_state=95)
        
        # fit model
        treeClf = treeClf.fit(xtrain, ytrain)
        
        # train accurecy score
        train_score = treeClf.score(xtrain, ytrain)
        validate_score = treeClf.score(xvalidate, yvalidate)
        
        # create a dictionary of scores
        output = {
            "depth": d,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": train_baseline_acc_score - train_score,
            "validate_baseline_diff": train_baseline_acc_score - validate_score,
            "baseline_accuracy": train_baseline_acc_score,
        }
        
        metrics.append(output)

    # get the result as a dataframe
    model_df = pd.DataFrame(metrics)

    # find the model with smallest ifference between the train and validate
    high_models = model_df[model_df.validate_score > model_df.baseline_accuracy + 0.05]

    bestModel = high_models[high_models.difference == abs(high_models.difference).min()]

    # return the best knn model
    return bestModel

def random_forest_model(xtrain = xtrain, ytrain = ytrain, xvalidate=xvalidate, yvalidate=yvalidate):
    """
    Goal: Train random forest model on different tree depth and return the best ferforming model
    """

    metrics = []
    for trees in range(2,20):
        # base line
        train_baseline_acc_score = accuracy_score(train.churn, train.baseline)
        
        # create ramdom tree object
        randFor = RandomForestClassifier(n_estimators= 100, min_samples_leaf= trees, max_depth = trees, random_state=95 )
        
        # fit the model
        randFor = randFor.fit(xtrain, ytrain)
        
        # get accuracy scores
        train_score = randFor.score(xtrain, ytrain)
        validate_score = randFor.score(xvalidate, yvalidate)
        
        # create a dictionary of scores
        output = {
            "trees": trees,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": train_baseline_acc_score - train_score,
            "validate_baseline_diff": train_baseline_acc_score - validate_score,
            "baseline_accuracy": train_baseline_acc_score,
        }
        
        metrics.append(output)

    # get the result as a dataframe
    model_df = pd.DataFrame(metrics)

    # find the model with smallest ifference between the train and validate
    high_models = model_df[model_df.validate_score > model_df.baseline_accuracy + 0.06]

    bestModel = high_models[high_models.difference == abs(high_models.difference).min()]

    # return the best knn model
    return bestModel

def logistic_regression_model(xtrain = xtrain, ytrain = ytrain, xvalidate=xvalidate, yvalidate=yvalidate):
    """
    Goal: Train logistic regression model on different C and return the best performing model
    """

    metrics = []

    for c in np.arange(0.0001,0.1, 0.001):
        # base line
        train_baseline_acc_score = accuracy_score(train.churn, train.baseline)
        
        # create ramdom tree object
        logReg = LogisticRegression(C=c, random_state=95, max_iter= 1000)
        
        # fit the model
        randFor = logReg.fit(xtrain, ytrain)
        
        # get accuracy scores
        train_score = randFor.score(xtrain, ytrain)
        validate_score = randFor.score(xvalidate, yvalidate)
        
        # create a dictionary of scores
        output = {
            "C": c,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": train_baseline_acc_score - train_score,
            "validate_baseline_diff": train_baseline_acc_score - validate_score,
            "baseline_accuracy": train_baseline_acc_score,
        }
        
        metrics.append(output)

    # get the result as a dataframe
    model_df = pd.DataFrame(metrics)

    # find the model with smallest ifference between the train and validate
    high_models = model_df[(model_df.C > 0.045) & (model_df.C < 0.055)]

    bestModel = high_models[high_models.difference == high_models.difference.min()]

    # return the best knn model
    return bestModel


def best_model_test(xtrain = xtrain, ytrain = ytrain, xTest=xTest):
    """
        Goal: test logistic regression model on the best C on my model and return the accuracy of test.
        Also write the test predictions in a csv file.
    """
    # base line
    train_baseline_acc_score = accuracy_score(train.churn, train.baseline)

    # create ramdom tree object
    logReg = LogisticRegression(C=0.0471, random_state=95, max_iter= 1000)

    # fit the model
    randFor = logReg.fit(xtrain, ytrain)

    # pretict test
    y_test_pred = logReg.predict(xTest)

    # get prediction probability
    y_test_pred_proba = logReg.predict_proba(xTest)

    # get only probability of churn
    churn_proba = []
    for i in y_test_pred_proba:
        churn_proba.append(i[1])
        
    # ouput for csv
    output = {
        
        "customer_id": test.customer_id,
        "churn_proba": churn_proba,
        "prediction": y_test_pred
    }

    # # get accuracy scores
    # train_score = randFor.score(xtrain, ytrain)
    # validate_score = randFor.score(xvalidate, yvalidate)
    test_score = randFor.score(xTest, yTest)

    # pandas dataframe to convert to csv
    customer_predictions = pd.DataFrame(output)

    # covert to csv
    customer_predictions.to_csv("customer_prediction.csv", mode='w')

    return f"Logistic regression test accuracy: {round(test_score, 2)}"

