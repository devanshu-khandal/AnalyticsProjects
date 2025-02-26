## Linear Regression Practice
import warnings
warnings.filterwarnings('ignore')

#%%
import numpy as np
import pandas as pd
# from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
# from sklearn.datasets import clear_data_home
# clear_data_home()  
import matplotlib.pyplot as plt

# In[1]:
# df = pd.read_csv("/Users/devanshu_khandal/Analytics_Projects/ML_DataSets/CaliforniaHousing/cal_housing.data"
#                 ,header=None)

df=fetch_california_housing()

df.feature_names

dataset = pd.DataFrame(df.data, columns=df.feature_names)

dataset.columns = df.feature_names 
dataset.head()
# help(pd.read_csv)
# dataset = pd.DataFrame(data= np.c_[df['data'],df['target']],
#                      columns= df['feature_names'] + ['target'])
# dataset.head()

# In[2]:
## Independent and Dependent features

X = dataset
y = df.target

# In[3]:
## Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=0.3, random_state=0)

# In[4]:
## Implement Linear Regression
from sklearn.linear_model import LinearRegression

# we have to standardize the data (only Independent features and that too only train and test data)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#to Inverse use scaler.inverse_transform(X_train)

# In[5]:
## Implement Linear Regression
regressor = LinearRegression()

# we will use cross validation to find the best model
from sklearn.model_selection import cross_val_score
mse=cross_val_score(regressor,X_train,y_train,scoring='neg_mean_squared_error',cv=5) #we will use negative mean squared error

# %%
mse.mean() # shold be almost 0 or least possible

# In[6]:
## Predicting the Test set results
regressor.fit(X_train, y_train)
reg_pred = regressor.predict(X_test)

# %%
## Plotting the graph to check with truth values
import seaborn as sns
sns.distplot(y_test-reg_pred,kde=True)

# %%
# R2 Score
from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)
score

# In[7]:
## Implement Ridge Regression to handle overfitting (l2 regularization) (lambda*sum(wi^2)) where 
# lambda is alpha and wi is weight (slope)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV # to find the best alpha value (cross validation)

ridge_regressor = Ridge()
parameters = {'alpha':[1,2,5,10,20,30,40,50,60,70,80,90,100]}
ridgecv = GridSearchCV(ridge_regressor, parameters, scoring='neg_mean_squared_error', cv=5)
ridgecv.fit(X_train, y_train)

print(ridgecv.best_params_)
print(ridgecv.best_score_)
print(mse.mean())
ridge_pred=ridgecv.predict(X_test)

# %%
## Plotting the graph to check with truth values
sns.distplot(ridge_pred-y_test,kde=True)

# In[8]:
## Implement Lasso Regression for automatic selection of features along with handling overfitting 
# (l1 regularization) (lambda*sum(|wi|)) where
# lambda is alpha and wi is weight (slope)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso_regressor = Lasso()
parameters = {'alpha':[1,2,5,10,20,30,40,50,60,70,80,90,100]}
lassocv = GridSearchCV(lasso_regressor, parameters, scoring='neg_mean_squared_error', cv=5)
lassocv.fit(X_train, y_train)

print(lassocv.best_params_)
print(lassocv.best_score_)

lasso_pred=lassocv.predict(X_test)
sns.distplot(lasso_pred-y_test,kde=True) # to check the difference between predicted and actual values 

# we will use Ridge Regression as it is giving better results

# In[9]:
## Saving the model to reuse it
# import pickle
# lasso_pred

## Logistic Regression

# In[10]:
from sklearn.linear_model import LogisticRegression
dd = sns.load_dataset('iris')
dd.head()
dd.isnull().sum()
dd['species'].value_counts()

dd=dd.loc[~dd['species'].str.contains('setosa')]

# In[11]:
## Independent and Dependent features
dd['species']=dd['species'].map({'versicolor':1,'virginica':0})
X = dd.iloc[:,:-1]
y = dd.iloc[:,-1]

# In[12]:
## Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.3, random_state=42)

# In[13]:
## Implement Logistic Regression
classifier = LogisticRegression()

from sklearn.model_selection import GridSearchCV # to find the best alpha value (cross validation) that is hyperparameter tuning
parameters={'penalty':['l1','l2','elasticnet'],'C':[0.001,0.01,0.1,1,10,100,1000],
            'max_iter':[100,200,300]}

classifier_regressor=GridSearchCV(classifier,parameters,scoring='accuracy',cv=5) # for classification we use accuracy not neg_mean_squared_error
classifier_regressor.fit(X_train,y_train)

print(classifier_regressor.best_params_)
print(classifier_regressor.best_score_)

classifier_pred=classifier_regressor.predict(X_test)

# In[14]:
## accuracy score
from sklearn.metrics import accuracy_score,classification_report
score = accuracy_score(y_test, classifier_pred)
print(score)
print(classification_report(y_test,classifier_pred))

# we can use this model to predict the species of the flower based on the features

# In[15]:
## EDA (feature selection or feature engineering)
sns.pairplot(dd,hue='species')
dd.corr()

