import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('diabetes.csv')
df = df.rename(columns={'ds': 'DS'})


columns = ['ID', 'No_Pation']
df.drop(columns, inplace=True, axis=1)
df['Gender'].replace({'M': 1, 'F': 0, 'f': 0}, inplace=True)
df['Gender'].astype('int')

df_copy = df.copy(deep=True)
df_copy = df.copy(deep=True)
df_copy[['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']] = df_copy[[
    'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['AGE'].fillna(df_copy['AGE'].mean(), inplace=True)
df_copy['Urea'].fillna(
    df_copy['Urea'].median(), inplace=True)
df_copy['Cr'].fillna(df_copy['Cr'].median(), inplace=True)
df_copy['HbA1c'].fillna(df_copy['HbA1c'].median(), inplace=True)
df_copy['Chol'].fillna(df_copy['Chol'].median(), inplace=True)
df_copy['TG'].fillna(df_copy['Chol'].median(), inplace=True)
df_copy['HDL'].fillna(df_copy['HDL'].median(), inplace=True)
df_copy['LDL'].fillna(df_copy['LDL'].median(), inplace=True)
df_copy['VLDL'].fillna(df_copy['VLDL'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)


def print_estimator_name(estimator):
    return estimator.__class__.__name__


estimators = []  # estimators array
# classifiers array to save classification models
classifiers = []
scores = []  # scores array for saveing presicion and recall results for comparing
acc = []  # saving accuracy scores
f1 = []  # saveing f1 score


# Model Building
X = df.drop(columns='CLASS')
y = df['CLASS']
y.replace({'Y': 1, 'N': 0, "P": 2, 'N ': 0, 'Y ': 1}, inplace=True)
y = y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)


# Creating DS Model

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
cm = metrics.confusion_matrix(y_test, y_pred)

df.head()
# describe a dfFream and info
df.info()
df.describe()

print(df)

print('accuracy', accuracy*100)
print('precision', precision*100)
print('recall', recall*100)
print('f1_score', f1_score*100)
print('confusion matrix \n', cm)

classifiers.append(classifier)
scores_dict = {"precision": precision, "recall": recall}
scores.append(scores_dict)
acc.append(accuracy)
f1.append(f1_score)
estimators.append(print_estimator_name(classifier))

print('estimators', estimators)
print('='*100)
print('classifiers', classifiers)

scores_df = pd.DataFrame(scores)
print(scores_df)


# Creating a pickle file for the classifier
joblib.dump(classifier, 'ds.pkl')
