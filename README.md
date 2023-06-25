# <a name="top"></a>Telc Churn Classification Project
![]()

by: Victoire Migashane

<p>
  <a href="https://github.com/MigashaneVictoire" target="_blank">
    <img alt="Victoire" src="https://img.shields.io/github/followers/MigashaneVictoire?label=Follow_Victoire&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

<!-- <img src="https://docs.google.com/drawings/d/e/2PACX-1vR19fsVfxHvzjrp0kSMlzHlmyU0oeTTAcnTUT9dNe4wAEXv_2WJNViUa9qzjkvcpvkFeUCyatccINde/pub?w=1389&amp;h=410"> -->

## <a name="project_description"></a>Project Objective:
[[Back to top](#top)]

- Refine your work into a final report, in the form of a jupyter notebook, that shows the work you did, why, goals, what you found, your methodologies, and your conclusions
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in your report (notebook)
- Create modules (acquire.py, prepare.py) that make your process repeateable and your report (notebook) easy to read and follow
- Ask exploratory questions of your data that will help you understand more about the attributes and drivers of customers churning (answer questions through charts and statistical tests)
- Construct a model to predict customer churn using classification techniques, and make predictions for a group of customers
- Walk through your report (notebook) in a 5 minute presentation with the lead data scientist
- Be prepared to answer panel questions about your code, process, findings and key takeaways, and model

**Goals:**
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that accurately predicts customer churn
- Present your process and findings to the lead data scientist

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
- Create all the files I will need to make a funcitoning project (.py and .ipynd files)
- Create a .gitignore file and ignore my env.py file
- Start by acquiring data from the codeup database and document all my initail acquisition steps in the acquire.py file
- Using the prepare file, clearn the data and split it into train, validatate, and test sets.
- Explore xplore the data. (Focus on the main main questions)
    - Does the type of internet service significantly impact the likelihood of churn?
    - What month are customers most likely to churn and does that depend on their contract type?¶
    - Do customers with technical support have a lower churn rate compared to those without it?¶

- Anser all the quesitons with statistical testing.
    - Identify drivers of churn.
- Make prediction of customer churn using driving features of churn.
- Document findings (include 4 visuals)
    - Add important finding to the final notebook
    - Create csv file of test predictions on best perfomring model.

        
### Questions

- Does the type of internet service significantly impact the likelihood of churn?
- What month are customers most likely to churn and does that depend on their contract type?
- Does the customer having a partner or dependents affect churn.¶
- Do customers who churn have a higher average monthly spend than those who don't?
- Does the presence or absence of phone service influence churn rates?
- Does the paperless billing option affect the likelihood of churn?
- Do customers with technical support have a lower churn rate compared to those without it?¶

### Target variable

- The target variable is the `churn` column in the data.

### Need to haves (Deliverables):

**1. Readme (.md)**

- project description with goals
- initial hypotheses and/or questions you have of the data, ideas
- data dictionary
- project planning (lay out your process through the data science pipeline)
- instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
- key findings, recommendations, and takeaways from your project

**2. Final Report (.ipynb)**

- A Report that has filtered out all the extraneous elements not necessary to include in the report.
- Use markdown throughout the notebook to guide the audience. Assume the reader will not read your code blocks as you think about how much markdown guidance do you need.
- Then, assume another reader will read ALL of your code, so make sure it is very very clearly commented. All cells with code need comments.
- Your notebook should begin with a project overview and goals
- Preparation should specifically call out any ways you changed the data (like handling nulls)
- Provide the context of the target variable through a visualization (distribution of the values, e.g.)
- Exploration should be refined in the report because now you know which visualizations and tests led to valuable outcomes.
- Include at least 4 visualizations in the form of:
    - Question in markdown that you want to answer
    - Visualization
    - Statistical test (in at least 2 of your 4)

    - Provide your clear answer or takeaway in markdown and natural language to the question based on your exploration

- Include your 3 best models in the final notebook to review. Show the steps and code you went through to fit the models, evaluate, and select.
- On your best model, a chart visualizing how it performed on test would be valuable.
- End with a conclusion that talks about your original goals and how you reached those (or didn't), the key findings, recommendations and next steps ("If I had more time, I would...")

**3. Acquire & Prepare Modules (.py)**

- contains functions to acquire, prepare and split your data. You can have other .py files if you desire to abstract other code away from your final report.
- Your work must be reproducible by someone with their own env.py file.
- Each of your functions are complimented with docstrings. If they are functions you borrowed from instructors, put those docstrings in your own words.
- Functions to acquire and prepare your data should be imported and used in your final report.

**4. Predictions (.csv).**

- 3 columns: customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn).
- These predictions should be from your best performing model ran on X_test.
- Note that the order of the y_pred and y_proba are numpy arrays coming from running the model on X_test. The order of those values will match the order of the rows in X_test, so you can obtain the customer_id from X_test and concatenate these values together into a dataframe to write to CSV.

**5. non-final Notebook(s) (.ipynb)**

- There should be at least 1 non-final notebook
- these were created while working on the project, containing exploration & modeling work (and other work), not shown in the final report

### Nice to haves (With more time):

- Get dated data on monthly transactions.
- Get customer age data.
- Get geographic data.
- Data on customer standing debt with telco services.

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- Contract type, internet service type, monthly charges, tenure and  technical support seems to be strong drivers for predicting churn.

***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
|**Feature**|**Definition**|
|----|----|
|`payment_type_id`| Represents an identifier or code for different payment types, such as credit card, bank transfer, or electronic payment methods.|
|`gender`| Rrepresent the gender of the customer, indicated as "Male" or "Female".|
|`senior_citizen`| Indicate whether the customer is classified as a senior citizen, represented as a binary variable (1 for senior citizen, 0 for non-senior).|
|`partner`| Indicate whether the customer has a partner or spouse, represented as a binary variable (1 for yes, 0 for no).|
|`dependents`| Indicate whether the customer has dependents (e.g., children or other family members), represented as a binary variable (1 for yes, 0 for no).|
|`tenure`| Represents the length of time (in months) that a customer has been using the service or has been a subscriber.|
|`phone_service`| This column could indicate whether the customer has phone service, represented as a binary variable (1 for yes, 0 for no)."
|`tech_support`| Indicates whether the customer has technical support available for their service (e.g., 1 for yes, 0 for no).|
|`streaming_tv`| Indicates whether the customer has access to streaming television services (e.g., 1 for yes, 0 for no).|
|`streaming_movies`| Indicates whether the customer has access to streaming movie services (e.g., 1 for yes, 0 for no).|
|`paperless_billing`| Indicates whether the customer has opted for paperless billing (e.g., 1 for yes, 0 for no).|
|`monthly_charges`| Represents the monthly charges or fees for the service subscribed by the customer.|
|`total_charges`| Represents the total charges incurred by the customer during their tenure.|
|`churn`| Indicates whether the customer has churned or discontinued their service (e.g., 1 for churned, 0 for active).|
|`contract_type`| Represents the type of contract or service agreement (e.g., month-to-month, one-year contract, two-year contract).|
|`internet_service_type`| Represents the type of internet service (e.g., DSL, fiber optic, cable).|
|`payment_type`| Represents the payment method used by the customer (e.g., credit card, bank transfer, electronic payment).|

***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

- **Teclco churn data from Codeup database.**

Query: get access using the env.py file 

~~~
SELECT *
FROM customers #payment_types
JOIN contract_types ct USING(contract_type_id)
JOIN internet_service_types ist USING(internet_service_type_id)
JOIN payment_types pt USING(payment_type_id);
~~~

**env.py setup**
~~~
# retrieve Codeup mySQL data
def get_db_access(database): # takes database name
    # login info
    hostname = "" # codeup data base info
    username = "" # username info
    password = "" # user pass-code  info
    
    # acces url
    acc_url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return acc_url
~~~


*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - prepare_.py  and .ipynb
    - explore_.py and .ipynb
    - modeling_.py and .ipynb


### Takeaways from exploration:

- The type of internet service plays a significant role in the likelihood of churn, with fiber optic customers as the main drivers for chrun.
- Customers are more likly to churn within the first 24 month and this all depend on the type of contract type they have, and  how much they pay monthly.
- Customers who receive technical support have higer probability to not churn than those who don't.

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Chi-Squred Test

**Does the type of internet service significantly impact the likelihood of churn?¶**


#### Hypothesis:

- $H_0$: The type of internet service does not significantly impact the likelihood of churn?
- $H_a$: The type of internet service significantly impact the likelihood of churn?

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:

- We have enough evidence to reject the null

#### Summary:

- Fiber optic customers are churing at a higer rate than expected. Telco is lossing about 280 customers than expected.

### Stats Test 2: 
    - Pearman's R
    - T-test one sample
    - Chi-squared test

**What month are customers most likely to churn and does that depend on their contract type?**

    - is there a linear relationship between monthly charges and tenure.
    - Are customers more or less likely to churn within the first 24 month with Telco.
    - To answer the main question in reguards to contract type.

#### Hypothesis:

**a**

- $H_0$: There is not linear relationship between monthly charges and tenure.
- $H_a$: There is linear relationship between monthly charges and tenure.

**b**

- $H_0$: Customer churn rate with in the first 24 month is > 50%
- $H_a$: Customer churn rate with in the first 24 month is <= 50%

**c**

- $H_0$: The churn month is independent (no association) of the contract type.
- $H_a$: The churn month is dependent (association) of the contract type.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:

**a**

- We have enough evidence to reject the null

**b**

- We fail to reject the null at this time

**c**

- We have enough evidence to reject the null

#### Summary:

**a**

- - The test rejects the null showing that ther is a very small linear relationship (dependence) between monthly charges and tenure.
- The visual shows that high numer of people churn with in the first 24 month with Telco.

**b**

- Test fails to reject the null. Customer churn rate with in the first 24 months looks to be more than 50%

**c**

- Tests show that we have enough evidence to say that customers are more likly to churn withn the first 24 month and indeed it depends on contract type.

**Note: There is 5 more statistical test performed and are all in the stats test fiel**

***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:

### Baseline
    
- Baseline Results: 0.734675
    

- Selected features for modeling:

```features = [Internet_service_type, monthly_charges, tenure, contract_type, phone_service, paperless_billing, and tech_support]```

    **Note: Only use the dummie (encoded) variables for the above features**

***

### Models and R<sup>2</sup> Values:
- Will run the following Classification models:
    - K-Nearest Neighbor
    - Decision Tree
    - Random Forest
    - Logistic Regression (Regression model)

## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

|**Model**|**Validation Score**|
|----|----|
|Baseline|0.734675|
|K-Nearest Neighbor|0.79489|
|Decision Tree|0.785664| 
|Random Forest|0.796309|
|Logistic Regression|0.798439|


- Logistic Regression model performed the best


## Testing the Model

- I selected logistic regression for my as my best model because it had more consistant results. And after running it through the test data, it performed just as good as the validation and test cases with an 80% accuracy score.

***

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]

- It would wise for telco to first direct our efforts on the month-to-month customer who have high monthly charges and are a within there first two years with telco's services. 
- Telco should try and reachout to these custmers and provide technical suport as needed and see if they would stay with the services if they could change to DSL or get a bonus for signing a 1-year or 2-year contract.
