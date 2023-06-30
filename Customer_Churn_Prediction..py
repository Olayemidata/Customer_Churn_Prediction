#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, auc, plot_roc_curve,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[3]:


#read in the data

df = pd.read_excel(r"C:\Users\USER\Desktop\...xlsx")
df.head()


# In[4]:


#separate the data into train and test sets

train,test = train_test_split(
    df, test_size=0.2, 
    random_state=123)


# In[5]:


#reset index

train.reset_index(drop=True, inplace = True)
train


# In[6]:


#drop columns that are not needed

train.drop(
        ['comp_phone_no','orders_phone_no','order_id','order_reference','date_updated','date_ordered','order_status','actual_orders','payment_date','discount_id'],
         axis=1, inplace=True
         ) 


# In[7]:


#Convert date columns to datetime format

#train['date_ordered']=pd.to_datetime(train['date_ordered'])
train['last_date_ordered']=pd.to_datetime(train['last_date_ordered'])


# In[8]:


train.head()


# In[9]:


#define features and target

X_train = train.drop(['churn'],axis=1)
y_train = train['churn']


# In[10]:


#create a function to convert object variables to categories so as to convert to numerical values(encoding)

X_train_num = X_train

for col_name in X_train_num.columns:
    if(X_train_num[col_name].dtype == 'object', 'datetime64'):
        X_train_num[col_name] = X_train[col_name].astype('category')
        X_train_num[col_name] = X_train[col_name].cat.codes
        
X_train.head()


# In[11]:


#create a list of the features

cols = ['num_of_comp','product_type',
                'payment_type','payment_trial',
              'is_fraudulent','last_date_ordered', 'discount_value'
                ]


# In[12]:


#scale data 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[13]:


#put the features in a dataframe and check the result

features = pd.DataFrame(X_train, columns=cols)
features


# In[14]:


features.describe()


# In[15]:


#check the correlation of the features

plt.figure(figsize=(12,10))
corre = features.corr()
sns.heatmap(corre, annot= True, cmap="YlGnBu")


# ### FEATURE IMPORTANCE

# In[16]:


#check the relevance of the features

decision_tree = DecisionTreeClassifier(max_depth = 10)
decision_tree.fit(X_train, y_train)
predictors = cols

coef = pd.Series(decision_tree.feature_importances_, predictors).sort_values(ascending=False)
coef.to_frame()


# #### Repeat above steps for test data

# In[17]:


#reset index

test.reset_index(drop=True, inplace = True)
test


# In[18]:


#drop columns that are not needed

test.drop(
        ['comp_phone_no','orders_phone_no','order_id','order_reference','date_updated','date_ordered','order_status','actual_orders','payment_date','discount_id'],
         axis=1, inplace=True
         ) 


# In[19]:


#Convert date columns to datetime format

#test['date_ordered']=pd.to_datetime(train['date_ordered'])
test['last_date_ordered']=pd.to_datetime(test['last_date_ordered'])


# In[20]:


test.head()


# In[21]:


#define features and target

X_test = test .drop(['churn'],axis=1)
y_test  = test ['churn']


# In[22]:


#create a function to convert object variables to categories so as to convert to numerical values(encoding)

X_test_num = X_test

for col_name in X_train_num.columns:
    if(X_test_num[col_name].dtype == 'object', 'datetime64'):
        X_test_num[col_name] = X_test[col_name].astype('category')
        X_test_num[col_name] = X_test[col_name].cat.codes
        
X_test.head()


# In[23]:


#scale and fit data

X_test = scaler.fit_transform(X_test)


# Model building

# In[24]:


#Write a function for multiple model selection

def model_to_use(input_ml_algo):
    if input_ml_algo == 'DT':
        model = DecisionTreeClassifier()
    elif input_ml_algo == 'RF':
        model = RandomForestClassifier()
    elif input_ml_algo == 'XGBC':
        model = XGBClassifier()
    elif input_ml_algo == 'LGBMC':
        model = LGBMClassifier()
    elif input_ml_algo=='LR':
        model=LogisticRegression()
    return model


# In[25]:


#Write a function to evaluate the performance of the models
#print the AUC curve as well as the confusion matrix table

def performance(model,X_train,y_train,X_test, y_test):
    y_pred = model.predict(X_test)

    # Predict probability for test dataset
    y_pred_prob = model.predict_proba(X_test)
    y_pred_prob = [x[1] for x in y_pred_prob]

    disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, 
    cmap='Blues', values_format='d', 
    display_labels=['active','churned']
 )

    print("\n Accuracy Score : \n ",accuracy_score(y_test,y_pred))
    print("\n AUC Score : \n", roc_auc_score(y_test, y_pred_prob))
    print("\n Confusion Matrix : \n ",confusion_matrix(y_test, y_pred))
    print("\n Classification Report : \n",classification_report(y_test, y_pred))

    print("\n ROC curve : \n")
    sns.set_style("white")
    plot_roc_curve(model, X_test, y_test)
    plt.show()
    


# In[27]:


#Train models
model = model_to_use("LR")
model.fit(X_train, y_train)
performance(model,X_train,y_train,X_test, y_test)


# In[28]:


model = model_to_use("RF")
model.fit(X_train, y_train)
performance(model,X_train,y_train,X_test, y_test)


# In[29]:


model = model_to_use("DT")
model.fit(X_train, y_train)
performance(model,X_train,y_train,X_test, y_test)


# In[30]:


#Train Logistic xgboost model and evaluate
model = model_to_use("XGBC")
model.fit(X_train, y_train)
performance(model,X_train,y_train,X_test, y_test)


# In[31]:


model = model_to_use("LGBMC")
model.fit(X_train, y_train)
performance(model,X_train,y_train,X_test, y_test)


# In[ ]:




