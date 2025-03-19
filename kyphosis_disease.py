import pandas as pd

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

Kyphosis_df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/kyphosis.csv')

print(Kyphosis_df)

print(Kyphosis_df.head())

print(Kyphosis_df.tail())

print(Kyphosis_df.info())

#List the average, minimum and maximum age (in years) considered in this study using two different methods
print(Kyphosis_df['Age'].mean())#age in months
print(Kyphosis_df['Age'].mean()/12)#age in years
print(Kyphosis_df['Age'].min()/12)
print(Kyphosis_df['Age'].max()/12)
print(Kyphosis_df.describe())

print(Kyphosis_df)
#PERFORM DATA VISUALIZATION
from sklearn.preprocessing import LabelEncoder

LabelEncoder_y = LabelEncoder()
Kyphosis_df['Kyphosis'] = LabelEncoder_y.fit_transform(Kyphosis_df['Kyphosis'])

print(Kyphosis_df)

Kyphosis_True = Kyphosis_df[Kyphosis_df['Kyphosis']==1]
Kyphosis_False = Kyphosis_df[Kyphosis_df['Kyphosis']==0]

print( 'Disease present after operation percentage =', (len(Kyphosis_True) / len(Kyphosis_df) )*100,"%")

import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
correlation_matrix = Kyphosis_df.corr()

# Drop "Kyphosis" from both rows and columns
correlation_matrix = correlation_matrix.drop(index="Kyphosis", columns="Kyphosis")

# Plot the heatmap without "Kyphosis"
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Drop the "Kyphosis" column
df_without_kyphosis = Kyphosis_df.drop(columns=["Kyphosis"])

# Generate the pair plot without "Kyphosis"
sns.pairplot(df_without_kyphosis)

plt.show()

#Plot the data countplot showing how many samples belong to each class
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming 'df' is your existing DataFrame and 'Kyphosis' is the target column
plt.figure(figsize=(6, 4))  # Set figure size for clarity
# Countplot for the 'Kyphosis' column
sns.countplot(x='Kyphosis', data=Kyphosis_df, palette=['#4C72B0', '#55A868'])
# Adding labels and title
plt.xlabel('Kyphosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Kyphosis Class Distribution', fontsize=14)
# Annotating the bars with counts
for p in plt.gca().patches:
    plt.gca().annotate(
        f'{p.get_height()}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center', va='bottom', fontsize=11
    )
plt.show()

#CREATE TESTING AND TRAINING DATASET/DATA CLEANING
# Let's drop the target label coloumns# Dropping the 'Kyphosis' column to create features (X) and target (y)
x = Kyphosis_df.drop('Kyphosis', axis=1)
y = Kyphosis_df['Kyphosis']
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

#TRAIN A LOGISTIC REGRESSION CLASSIFIER MODEL
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
# Logistic Regression Model
model = LogisticRegression()
# Training the model
model.fit(x_train, y_train)

#EVALUATE TRAINED MODEL PERFORMANCE
from sklearn.metrics import classification_report, confusion_matrix
# Predicting the Test set results
y_predict_test = model.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True, fmt=".2f")
plt.show()

print(classification_report(y_test, y_predict_test))

#IMPROVE THE MODEL by random forest
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
# Predicting the Test set results
y_predict_test = decision_tree.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test, y_predict_test))

feature_importances = pd.DataFrame(decision_tree.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)

#Train a random forest classifier model and assess its performance
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest.fit(x_train, y_train)
# Predicting the Test set results
y_predict_test = RandomForest.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test, y_predict_test))