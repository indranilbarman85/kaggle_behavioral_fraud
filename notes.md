# Data Exploration


## ** Obejective**

- score each transaction as frauduent or not fraudulent based on deviation from past behavior. 

## ** Evaluation Metric** 

- ROC-AUC on a hiddent test set..


## High level summary of data
### train

- train data has 182,125 transactions for 5,000 customers
- train data is **highly imbalanced** with only **1.64%** positive rate.
- There are 20 distinct **merchants**
- transactions from 30 distinct **countries**.
- **timestamp** is sequence number and not actual time, it just helps in ordering the transactions for each user.
- 8743 distinct **devices** so some customers have used more than 1 devices
- 2 distincr **channels**. Not sure what they are
- the **amount field** is **skewed** towards the **lower range** with 80% of amount in the range of 0-40. Break the 0-40 bucket and see the distribution in it. DO additional chheck to see the distribution of fraud cases in this bucket and outside the bucket.
- the "hours since pervious transaction" field is also skewed left with 60% of transactions are in the bucket 0-50000 bucket. Break the 0-50000 bucket and see the distribution.

## missing values

- both train and test datasets have no missing values.


## overlap with test

- all the users in train are in test.
- also seems like the data in test and train are overlapping timeline wise.
- I plan to merge the test train together and see


## combine datasets.    

## distribution

- customers have average of 46 transactions.
- 2233 / 5000 (44.66%) have at least one fraudulent transaction.

## univariate analysis

### Amount

- Skewed left
- fraud rate is higher at lower
- log transform for linear models. 
- max isn't extreme, so probably no hard capping needed.

### hours_since_prev_txn

- Skewed left
- no significant difference in distribution between fraudulent and non-fraudulent.


### merchant_category (20 distinct values)

- all merchants are uniformly distributed.
- fraud cases are also distributed uniformly.

### country (30 distinct values)

- similar trensds with country.