
# coding: utf-8
# In[1]:

from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

churn_df = pd.read_csv('../Data/train_set_anju.csv')
churn_df = churn_df.drop('cancellation_request',axis=1)

col_names = churn_df.columns.tolist()

for i, name in enumerate(list(churn_df.columns.values)):
        if churn_df[name].dtype in [int, float]:
            churn_df[name] = churn_df.values[:, i]
        else:
            encoder = LabelEncoder()
            churn_df[name] = encoder.fit_transform(
                [str(i) for i in churn_df.values[:, i]]) 
to_drop = ['id']
churn_df = churn_df.drop(to_drop,axis=1)

churn_result = churn_df['renewed']
y = churn_result.astype(int)
churn_df = churn_df.drop('renewed',axis=1)

churn_df['account_value'] = churn_df['account_value'].replace(np.nan, 0)
churn_df = churn_df.replace(np.nan, 100000)

to_drop = ['job_count_over_client_count', 'job_count_over_invoice_count', 'job_count_over_quote_count', 
           'client_count_over_job_count','client_count_over_invoice_count', 'client_count_over_quote_count',
          'invoice_count_over_job_count', 'invoice_count_over_client_count', 'invoice_count_over_quote_count',
          'quote_count_over_job_count', 'quote_count_over_client_count', 'quote_count_over_invoice_count']

print(any(churn_df.isnull()))
churn_df = churn_df.fillna(0)
print(any(churn_df.isnull()))
churn_df[:]


# Ways to Preprocess Data (Brainstorming):
# 1. Put country/plan_code in distribution of most to least common...something with the mode 
# 2. 

# In[2]:

pd.isnull(churn_df).any(1).nonzero()[0]


# In[3]:

#Convert data to np array 
X = churn_df.as_matrix()
poly = PolynomialFeatures(2) 
X = poly.fit_transform(X)


# In[4]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels:", np.unique(y))


# In[ ]:

from sklearn.model_selection import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for i, (train_index, test_index) in enumerate(KFold(n_splits=8).split(churn_df)):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
       
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


# In[ ]:

from sklearn.svm import SVC

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

for i in range(10):
    poly = PolynomialFeatures(i) 
    X = poly.fit_transform(X)
    print("i: %.3f" %i,  accuracy(y, run_cv(X,y,SVC)))


# In[ ]:



