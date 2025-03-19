#import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)

#import data
diabetes = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/diabetes.csv')

#printing database
print(diabetes)
print(diabetes.head(7))
print(diabetes.tail())
print(diabetes.info())

#average bmi,max,min average age
print(diabetes.describe())

#data visualization
plt.figure(figsize = (12, 7))
sns.countplot(x = 'Outcome', data = diabetes)
plt.show()
sns.pairplot(diabetes, hue = 'Outcome', vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
plt.show()
print(diabetes)

#split data nd prepare for training
x = diabetes.iloc[:,0:8].values
print(x)
y = diabetes.iloc[:, 8].values
print(y)

# feature scaling is must in ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print (x_train.shape)
print (x_test.shape)
print("split ratio 25% for testing nd 75% to training")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)
print (x_train.shape)
print (x_test.shape)

import tensorflow as tf

ANN_model = tf.keras.models.Sequential()
ANN_model.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(8, )))
ANN_model.add(tf.keras.layers.Dropout(0.2))

ANN_model.add(tf.keras.layers.Dense(units=400, activation='relu'))
ANN_model.add(tf.keras.layers.Dropout(0.2))

ANN_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ANN_model.summary()

#compile and train ANN model
ANN_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
epochs_hist = ANN_model.fit(x_train, y_train, epochs = 200)
y_pred = ANN_model.predict(x)
print(y_pred)

y_pred = (y_pred > 0.5)
print(y_pred)

#evaluate trained model performance
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss'])
plt.show()

plt.plot(epochs_hist.history['accuracy'])
#plt.plot(epochs_hist.history['val_accuracy'])
plt.plot(epochs_hist.history['loss'])
#plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# Make predictions on the test set
y_pred = ANN_model.predict(x_test)

# Convert predictions to binary (0 or 1)
y_pred = (y_pred > 0.5)

#trainig set performance
from sklearn.metrics import confusion_matrix

#testing set performance
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, annot = True)
plt.show()

# Calculate the confusion matrix using the test set predictions and true labels
cm = confusion_matrix(y_test, y_pred)
plt.title('Confusion Matrix During Training')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#trainig set performance
from sklearn.metrics import confusion_matrix
#testing set performance
cm = confusion_matrix(y_test, y_pred)
plt.title('Confusion Matrix During Training')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


import seaborn as sns
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix During Training')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()



import sklearn.metrics

print(sklearn.metrics.classification_report(y_test, y_pred))
import xgboost
# Train an XGBoost classifier model

import xgboost as xgb
XGB_model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 10)
XGB_model.fit(x_train, y_train)
result_train = XGB_model.score(x_train, y_train)
print("Accuracy train : {}".format(result_train))
# predict the score of the trained model using the testing dataset

result_test = XGB_model.score(x_test, y_test)
print("Accuracy test : {}".format(result_test))
# make predictions on the test data
y_predict = XGB_model.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_predict))
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)
plt.ylabel('Predicted class')
plt.xlabel('Actual class')
plt.title('Confusion Matrix During Training')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
from sklearn.linear_model import LogisticRegression
#Train a logistic regression classifier model and assess its performance
from sklearn.metrics import classification_report, confusion_matrix

model_LR = LogisticRegression()

model_LR.fit(x_train, y_train)

# predict the score of the trained model using the testing dataset

result_test = model_LR.score(x_test, y_test)
print("Accuracy : {}".format(result_test))

# make predictions on the test data
y_predict = model_LR.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot=True)
plt.ylabel('Predicted class')
plt.xlabel('Actual class')
plt.title('Confusion Matrix During Training')
plt.xlabel('Actual')
plt.ylabel('Predicted Class')
plt.show()
