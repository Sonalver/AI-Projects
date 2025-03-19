#import libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#load data on dataframe
df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/breast_cancer.csv')
#display dataframe
print(df.head())

#count of rows and columns
print(df.shape)
#count number of null (empty) values
print(df.isnull().sum())

# drop the colums with null values
df.dropna(axis=1,inplace=True)
#count rows and columns
print(df.shape)

#count of number of M annd B cellsin diagnosis
print(df['diagnosis'].value_counts())
#Lable Encoding
#Get datatypees of each column in dataset
print(df.dtypes)

from sklearn.preprocessing import LabelEncoder
# Assuming df is your DataFrame and you are working on the second column (index 1)
labelencoder = LabelEncoder()
# Transform the values in the second column
encoded_labels = labelencoder.fit_transform(df.iloc[:, 1].values)
# Replace the column in the DataFrame (if needed)
df.iloc[:, 1] = encoded_labels
print(f"Encoded Labels: {encoded_labels}")

# Optionally print the full DataFrame
df['Category'] = encoded_labels
print("Updated DataFrame:")
print(df)

#Split dataset and feature scaling
#spliting the dataset into independent and depedent datasets
x=df.iloc[:,2:].values
y=df.iloc[:,1].values

#spliting dataset into training(75%) and testing(25%)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#scaling the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#print data
print(x_train)

#Building logistic regression model
print(y_train)

 #Convert Y_train and Y_test to numeric as below:
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
y_train = y_train.astype(int)		#Y_train to numeric
y_test = y_test.astype(int)		        #Y_test to numeric
classifier.fit(x_train,y_train)
classifier.get_params()

#build a logistic regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#make use of trained model to make predtictions on test data
predictions = classifier.predict(x_test)

#plot confusion matrix
#plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test,predictions)
print(cm)
sns.heatmap(cm,annot=True)
plt.show()

#get accuracy score for model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

print(predictions)
