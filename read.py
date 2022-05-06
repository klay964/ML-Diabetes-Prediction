import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('diabets.csv')
df1 = df1.rename(columns={'ds': 'DS'})

columns = ['ID', 'No_Pation']
df1.drop(columns, inplace=True, axis=1)
df1['Gender'].replace({'M': 1, 'F': 0, 'f': 0}, inplace=True)
df1['Gender'].astype('int')

df_copy = df1.copy(deep=True)
df_copy = df1.copy(deep=True)
df_copy[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']] = df_copy[[
    'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Gender'].fillna(df_copy['Gender'].mean(), inplace=True)
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


# Model Building
X = df1.drop(columns='CLASS')
y = df1['CLASS']
y.replace({'Y': 1, 'N': 0, "P": 2, 'N ': 0, 'Y ': 1}, inplace=True)
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Creating DS Model

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
joblib.dump(classifier, 'ds.pkl')
