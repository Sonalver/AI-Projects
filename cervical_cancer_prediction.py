import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
#!pip install plotly
import plotly.express as px
# Install the missing jupyterthemes package
#!pip install jupyterthemes
from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)
# setting the style of the notebook to be monokai theme
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will n
cancer_df = pd.read_csv('../../resources/cervical_cancer.csv')
#cancer_df
cancer_df.tail(20)
cancer_df_last_20 = cancer_df.head(20) # Changed 'df' to 'cancer_df'
print(cancer_df_last_20) # Assuming you want to print the head, not a non-existent variable 'cancer_df_first_20'
last_20_rows = cancer_df.tail(20)
print(last_20_rows)
cancer_df.head(20)
#data analysis
# Get data frame info
cancer_df.info()
cancer_df.describe()
cancer_df
#get data frame info
cancer_df.info()
#get statistical of data frame
cancer_df.describe()
# Notice many question marks indicating missing values
cancer_df
#replace ? with nan
cancer_df = cancer_df.replace('?',np.nan)
cancer_df
#plot heat map
cancer_df.isnull()
plt.figure(figsize = (20,20))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.show()

cancer_df.info()
# Since STDs: Time since first diagnosis  and STDs: Time since last diagnosis have more than 80% missing values
# we can drop them
cancer_df = cancer_df.drop(columns = ['STDs: Time since first diagnosis','STDs: Time since last diagnosis'] )
cancer_df
# Since most of the column types are object, we are not able to get the statistics of the dataframe.
# Convert them to numeric type

cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df.info()
cancer_df.describe()
cancer_df.mean()
#replace null values with mean
cancer_df = cancer_df.fillna(cancer_df.mean())
cancer_df
#Nan heatmap
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.show()

#range
cancer_df['Age'].min()
cancer_df['Age'].max()
cancer_df[cancer_df['Age'] == 84]
corr_matrix = cancer_df.corr()
corr_matrix
plt.figure(figsize = (30,30))
sns.heatmap(corr_matrix, annot = True)
plt.show()

#plot histogram of entire data set
cancer_df.hist(bins = 10, figsize = (30,30), color = 'b')
plt.show()

#prepare data before training
cancer_df
target_df = cancer_df['Biopsy']
input_df = cancer_df.drop(columns = ['Biopsy'])
target_df.shape
input_df.shape
x = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')
#reshaping array
y.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x
#spliting data into test and trainig sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5)
#split data such as testing data is quater the size of training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
cancer_df
#pip install xgboost
#train an xgboost classifer model
import xgboost as xgb
#!pip install mxnet-cu92
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators =10)

model.fit(x_train, y_train)
result_train = model.score(x_train, y_train)
result_train
result_test =model.score(x_test, y_test)
result_test
y_predict = model.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_predict))
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)
plt.show()
import xgboost as xgb

model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 50, n_estimators =100)

model.fit(x_train, y_train)
result_train = model.score(x_train, y_train)
result_train
result_test =model.score(x_test, y_test)
result_test
y_predict = model.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_predict))
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)
plt.show()



