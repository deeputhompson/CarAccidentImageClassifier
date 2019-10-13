#!/usr/bin/env python
# coding: utf-8

# In[8]:
import numpy as np
import pandas as pd
# # DATASET
# In[9]:
claims_dataset=pd.read_csv(r"C:\Users\deepu\Desktop\iCAP\Work Directory\Training Dataset\Training Dataset.csv")
# # Exploratory Data Analysis
# In[10]:
claims_dataset.head()
# In[11]:
print(claims_dataset.shape)
# In[12]:
claims_dataset.isnull().sum()
# # Feature Engineering
# In[13]:
# In[29]:
claims_dataset=claims_dataset.set_index("policy_number")
claims_dataset=claims_dataset[list(filter(lambda x:  not x.find("date")>-1, claims_dataset.columns ))]
X,y=claims_dataset.iloc[:,:-1],claims_dataset.iloc[:,-1]
# # Data Transformation
# In[31]:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
numColList=(X.dtypes[(X.dtypes=="int64") | (X.dtypes=="float64")]).index
strColList=X.dtypes[X.dtypes=="object"].index
enc=make_column_transformer(
    (MinMaxScaler(), numColList),
    (OneHotEncoder(),strColList))
X_tranform=enc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tranform, y, random_state=0)
# # MODEL SELECTION
# In[32]:
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# In[33]:
clf = GradientBoostingClassifier(max_depth=1, n_estimators=5000, learning_rate =0.5).fit(X_train, y_train)
prediction_on_testData=clf.predict(X_test)
print("Test Accuracy", clf.score(X_test,y_test))
confusion_matrix(prediction_on_testData,y_test)
# In[190]:
#get_ipython().run_line_magic('pinfo', 'GradientBoostingClassifier')
# In[34]:
prediction_on_testData
# ### Prediction probabilities for positive and negative class
# In[35]
Positive_proba=clf.predict_proba(X_test)[:,1]
print("\n\nPositive Class Probability\n\n",Positive_proba)
Negative_proba=clf.predict_proba(X_test)[:,0]
print("\n\nNegative Class Probability\n\n", Negative_proba)
# In[36]:
pd.DataFrame(clf.predict_proba(X_test), columns=["proba_N", "proba_Y"])
# In[37]:
prediction_onb_trainingData=clf.predict(X_train)
print("Training Accuracy", clf.score(X_train,y_train))
confusion_matrix(prediction_onb_trainingData,y_train)
# In[38]:
clf.predict(X_test[6])
# ## Predictiction Activity, if we had a new record being flowing in:
# Transformation is applied to the new record for which prediction is to be done and then use  .predict function
# In[337]:
dfInputFeed= pd.read_csv('C:\\Users\\deepu\\Desktop\\iCAP\\Work Directory\\Model Input\\ClaimsFraud.csv')
# In[338]:
dfInputFeed.info()
print("Predicting Fraud for the Policy Number-"+str(dfInputFeed['policy_number'][0]))
print("Claim Incident Date is-"+str(dfInputFeed['incident_date'][0]))
# In[339]:
dfInputFeed=dfInputFeed.set_index("policy_number")
dfInputFeed=dfInputFeed[list(filter(lambda x:  not x.find("date")>-1, dfInputFeed.columns ))]
X=dfInputFeed
# In[340]:
''''numColList=(X.dtypes[(X.dtypes=="int64") | (X.dtypes=="float64")]).index
strColList=X.dtypes[X.dtypes=="object"].index
enc=make_column_transformer(
    (MinMaxScaler(), numColList),
    (OneHotEncoder(),strColList))
X_tranform=enc.fit_transform(X)'''
# In[341]:
X
# In[342]:
X.iloc[1:2]
# In[343]:
predValue = clf.predict(enc.transform(X.iloc[0:1]))[0]
print("Predicted value-"+predValue)
# In[329]:
# In[312]:
#numColList=(X.iloc[0:1].dtypes[(X.iloc[0:1].dtypes=="int64") | (X.dtypes=="float64")]).index
#strColList=X.iloc[0:1].dtypes[X.iloc[0:1].dtypes=="object"].index
#enc=make_column_transformer(
 #   (MinMaxScaler(), numColList),
  #  (OneHotEncoder(),strColList))
#X_tranform=enc.fit_transform(X.iloc[0:1])
#Positive_proba=clf.predict_proba(X.iloc[0:1])[:,1]
#Ngative_proba=clf.predict_proba(X.iloc[0:1])[:,0]
# In[313]:
#dfObj=pd.DataFrame([X.index[0]],[clf.predict(enc.transform(X.iloc[0:1]))[0]])
#dfObj=pd.DataFrame(X.index[0],clf.predict(enc.transform(X.iloc[0:1]))[0])
# In[314]:
#dfObj
# In[ ]:
# In[ ]:
# In[ ]:
