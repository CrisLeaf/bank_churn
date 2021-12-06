# Bank Churn


Bank (as well as streaming services) are often interested into calculating customear churn rate 
as the cost of retaining existing customers is far less than acquiring a new one.

In this analysis, we centered into finding the most relevant behaviors of those who leave the 
bank. And lately into building a Machine Learning model to predict the probability of churn of each 
customer, to focus the attention on those who are about to churn.


## Requirements

The Python Notebook `bank_churn_analysis.ipynb` was written in a Python 3.9 environment and using 
the following libraries:


- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Scipy](https://scipy.org/)
- [PrettyTable](https://pypi.org/project/prettytable/)



## Dataset

The dataset was downloaded from Kaggle website

- [Bank Churn](https://www.kaggle.com/mathchi/churn-for-bank-customers)

The data consists on the columns:
- `RowNumber`: the record number.
- `CustomerId`: random number to identify each customer.
- `Surname`: customers's Surname.
- `CreditScore`: customers's credit score.
- `Geography`: customer's location.
- `Gender`: customer's gender.
- `Age`: customer's age.
- `Tenure`: the number of year that the customer has been client of the bank.
- `Balance`: customer's balance.
- `NumOfProducts`: numbers of products that the customer has purchased.
- `HasCrCard`: if the customer has a creditcard. (0 for no, 1 for yes)
- `IsActiveMember`: if the customer is active. (0 for no, 1 for yes)
- `EstimatedSalary`: customer's estimated salary.
- `Exited`: if the customer have churn. (0 for no, 1 for yes)


## Methodology

We make an statistical, univariate, bivariate and multivariate analysis to get some insights of 
the data.

Lately, we evaluated various well known machine learning models. Taking into account differents 
metrics to give us a general idea on how the model is behaving.


## Results

We got a complete list of variables (and their percentage of impact) that affects whether a 
customer will leave.

Also we got a model capable of predicting with probability, when a customer is about to leave.


## Support

Give a :star: if you like it :hugs:.