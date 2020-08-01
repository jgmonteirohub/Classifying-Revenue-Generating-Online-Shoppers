# Import necessary librarys and modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

#Convert data csv into pandas dataframe 

data = pd.read_csv("online_shoppers_intention.csv")

# Read dataset

print(data.head())
print(data.describe())

# Data set contains a number of columns with null values that must be addressed later

# Data Visualization

# Proportion of Buying Customers

labels = data['Revenue'].unique()
counts = data["Revenue"].value_counts().values

plt.pie(counts,labels=labels,
               colors=["skyblue","Orange"],
               autopct='%1.1f%%',
               wedgeprops={'edgecolor':"Black"},
               labeldistance=None)
               
plt.legend(loc="upper left",bbox_to_anchor=(0.9, 0.95),shadow=True)
plt.title("Did the Visitor Generate Revenue?")
plt.show()

# Revenue on Weekends vs Revenue on Weekdays

data2 = pd.crosstab(data["Weekend"],data["Revenue"])
data2.plot(kind="bar", stacked=True,color=["skyblue","Orange"])
plt.show()

# Revenue by Month

data2 = pd.crosstab(data["Month"],data["Revenue"])
data2.plot(kind="bar", stacked=True,color=["skyblue","Orange"], sort_columns=True)
plt.show()

# Data Preparation

# Convert boolean/categorical data into dummy variables

data["Revenue"] = data["Revenue"].astype(int)
data["Month"] = pd.get_dummies(data["Month"])
data["VisitorType"] = pd.get_dummies(data["VisitorType"])

# Separate predictor and response variables

x = data.drop("Revenue",axis=1)
y = data["Revenue"]
features = x.columns

#Replace missing values with median of column data

print(data.isnull().sum()[data.isnull().sum()>0])

# There are 8 columns each with 14 null values. The null values will be replaced using each columns respective median.

categorical_columns = []
numeric_columns = []

for name in features:
    if x[name].map(type).eq(str).any(): 
        categorical_columns.append(name)
    else:
        numeric_columns.append(name)

x_numeric = data[numeric_columns]
x_categorical = pd.DataFrame(data[categorical_columns])

imp = SimpleImputer(missing_values=np.nan, strategy='median')
x_numeric = pd.DataFrame(imp.fit_transform(x_numeric), columns = x_numeric.columns)

x = pd.concat([x_numeric,x_categorical],axis=1)

# Scale data

scaled = StandardScaler()
x_scaled = scaled.fit_transform(x)

# Split into training and test sets

X_train, X_test, y_train,y_test = train_test_split(x_scaled,y,train_size = 0.7,random_state=42)

# Hyperparameter Tuning

# Logistic Regression

lr_model = LogisticRegression()

param_grid= {
    "C": np.logspace(-3,3,7), 
    "penalty": ["l1","l2","elasticnet","none"]
    }

lr_cv=GridSearchCV(lr_model,param_grid,cv=10)
lr_cv.fit(X_train,y_train)

print("Best Parameters) ",lr_cv.best_params_)
print("Accuracy of Best Paramters:",lr_cv.best_score_)

# Decision Tree
tree_model = DecisionTreeClassifier()

param_grid = {
    'max_leaf_nodes': list(range(2, 100)), 
    'min_samples_split': [2, 3, 4]
    }

dt_cv = GridSearchCV(DecisionTreeClassifier(), param_grid, verbose=1, cv=10)
dt_cv.fit(X_train, y_train)

print("Best Parameters) ",dt_cv.best_params_)
print("Accuracy of Best Paramters:",dt_cv.best_score_)

# Random Forest
rf_model = RandomForestClassifier()

param_grid = { 
    'n_estimators': [25,50,75,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rf_cv = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10 )
rf_cv.fit(X_train,y_train)

print("Best Parameters) ",rf_cv.best_params_)
print("Accuracy of Best Paramters:",rf_cv.best_score_)

# Gradient Boosting

gb_model =GradientBoostingClassifier()

param_grid = {
    'n_estimators': [25,50,75,100],
    'max_features': ['auto','sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    "criterion": ["friedman_mse"]
    }

gb_cv = GridSearchCV(gb_model, param_grid, cv=10, n_jobs=-1)
gb_cv.fit(X_train,y_train)

print("Best Parameters) ",gb_cv.best_params_)
print("Accuracy of Best Paramters:",gb_cv.best_score_)

# Tuning the Gradient Boosting model's paramaters can be computational expensive which limits how much tuning can be done
# Results indicate that the Random Forest and Gradient Boosting models are the best performers

# Cross validation and test set evaluation

# Logistic Regression 

logreg = LogisticRegression(C=0.001,penalty='none')
logreg.fit(X_train,y_train)
log_pred = logreg.predict(X_test)
print("Logistic Regression Classification Report:\n", classification_report(log_pred,y_test))


# Decision Tree

dtree = DecisionTreeClassifier(max_leaf_nodes=8,min_samples_split=2)
dtree.fit(X_train,y_train)
dt_pred = dtree.predict(X_test)
print("Decision Tree Classification Report:\n", classification_report(dt_pred,y_test))

# Random Forest

randomforest = RandomForestClassifier(criterion='entropy',max_depth=7,max_features='auto',n_estimators=100)
randomforest.fit(X_train,y_train)
rf_pred = randomforest.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(rf_pred,y_test))

# Gradient Boosting

gradboost = GradientBoostingClassifier(criterion='friedman_mse',max_depth=4,max_features='sqrt',n_estimators=75)
gradboost.fit(X_train,y_train)
gb_pred = gradboost.predict(X_test)
print("Gradient Boosting Classification Report:\n", classification_report(gb_pred,y_test))

# The Random Forest and Gradient Boosting models perform the best on the testing set as well. 
# We can conclude either the Random Forest or Gradient Boosting models are the ideal choices for this classification problem


