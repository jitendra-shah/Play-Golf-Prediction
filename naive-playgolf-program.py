import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

#Importing the dataset 
dataset = pd.read_csv('playgolf.csv')
dataset.PlayGolf.replace(('Yes', 'No'), (1, 0), inplace=True)

X = dataset.iloc[:,2:-2].values
y = dataset.iloc[:, 5].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classifier.fit(X_train, y_train)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

aranged_temp=np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01)
aranged_hum=np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)

X2, X3 = np.meshgrid(aranged_temp,aranged_hum)

plt.contourf(X2, X3, classifier.predict(np.array([X2.ravel(), X3.ravel()]).T).reshape(X2.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X2.min(), X2.max())
plt.ylim(X3.min(), X3.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend()
plt.show()

# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

aranged_temp=np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01)
aranged_hum=np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)


X2, X3 = np.meshgrid(aranged_temp,aranged_hum)

plt.contourf(X2, X3, classifier.predict(np.array([X2.ravel(), X3.ravel()]).T).reshape(X2.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X2.min(), X2.max())
plt.ylim(X3.min(), X3.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Testing set)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend()
plt.show()

# Explained mean squared error:
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f'% r2_score(y_test,y_pred))

#Predicting the user input data
t=int(input("Enter the current temperature:"))
h=int(input("Enter the current humidity:"))
arr=np.array([t,h]).reshape(1,2)
arr_pred = classifier.predict(arr)
print ("\n")
# Printing the predicted results
if arr_pred == 1:
    print ("Play Golf = Yes\n")
else:
    print ("Play Golf = No\n")
