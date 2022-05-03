import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('diabets.csv')


# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'diabetes': 'DIABETES'})

df1['Gender'] = df1['Gender'].replace('M', 0)
df1['Gender'] = df1['Gender'].replace('F', 1)
df1['CLASS'] = df1['CLASS'].replace('Y', 0)
df1['CLASS'] = df1['CLASS'].replace('N', 1)
df1['CLASS'] = df1['CLASS'].replace('P', 2)


# df1.replace({'M': 0, 'F': 1, 'f': 1, 'Y': 0, 'N': 1,
#             'n': 1, 'y': 0, 'p': 2, 'P': 2}, inplace=True)

df1.drop(columns='ID')
df1.drop(columns='No_Pation')

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']] = df_copy[[
    'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['AGE'].fillna(df_copy['AGE'].mean(), inplace=True)
df_copy['Urea'].fillna(
    df_copy['Urea'].median(), inplace=True)
df_copy['Cr'].fillna(df_copy['Cr'].median(), inplace=True)
df_copy['HbA1c'].fillna(df_copy['HbA1c'].median(), inplace=True)
df_copy['Chol'].fillna(df_copy['Chol'].median(), inplace=True)
df_copy['HDL'].fillna(df_copy['HDL'].median(), inplace=True)
df_copy['LDL'].fillna(df_copy['LDL'].median(), inplace=True)
df_copy['VLDL'].fillna(df_copy['VLDL'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building
df1.head()
# describe a df1Fream and info
df1.info()
df1.describe()

# X = df1.drop(columns='CLASS')
# y = df1['CLASS']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=0)

# # Creating Random Forest Model

# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)

# # Creating a pickle file for the classifier
# joblib.dump(classifier, 'ds.pkl')
# data=pd.read_csv('diabets.csv')
# data.drop(['ID','No_Pation'],axis=1,inplace=True)


# #Convert text values to numerical values
# def convert_text_to_num(x):
#   set_x=set(x)
#   index_w={}
#   w_index={}
#   for i,w in enumerate(set_x):
#     index_w[i]=w
#     w_index[w]=i
#   x=x.replace(w_index)
#   return x,w_index,index_w


# data.CLASS,CLASS_1_w_index,CLASS_1_w_index=convert_text_to_num(data.CLASS)
# data.Gender,Gender_1_w_index,Gender_1_w_index=convert_text_to_num(data.Gender)
# data = data.astype('int64')


# data.head()
# #describe a DataFream and info
# data.info()
# data.describe()


# X=data.iloc[:,:11] #Split the data  inputs(features (x))
# print('feauter are : ','\n' ,X)

# y=data['CLASS']#split the data output(target(y))
# print('target is:','\n' ,y)

# print(y)


# #defining estimators function with a array to save the models after learning for bars chart ploting
# def print_estimator_name(estimator):
#     return estimator.__class__.__name__

# estimators=[] #estimators array
# classifiers = [] #classifiers array to save classification models for confusion matrices ploting
# scores = [] #scores array for saveing presicion and recall results for comparing
# acc=[] #saving accuracy scores for bars chart ploting
# f1=[] #saveing f1 score for bars chart ploting

# x_normalizer=preprocessing.StandardScaler()
# x_norm=x_normalizer.fit_transform(X)

# x_train,x_test,y_train,y_test=model_selection.train_test_split(x_norm,y,test_size=0.33)
# print('x_train :',x_train.shape), print('y_train: ',y_train.shape)
# print('x_test :',x_test.shape), print('y_test :',y_test.shape)
# DS= DecisionTreeClassifier()
# DS.fit(x_train, y_train)

# y_pred = DS.predict(x_test)

# accuracy = metrics.accuracy_score(y_test, y_pred)
# precision = metrics.precision_score(y_test, y_pred,average='weighted')
# recall = metrics.recall_score(y_test, y_pred,average='weighted')
# f1_score = metrics.f1_score(y_test, y_pred,average='weighted')

# print('accuracy',accuracy*100)
# print('precision',precision*100)
# print('recall',recall*100)
# print('f1_score',f1_score*100)

# classifiers.append(DS)
# scores_dict = { "precision": precision, "recall": recall}
# scores.append(scores_dict)
# acc.append(accuracy)
# f1.append(f1_score)
# estimators.append(print_estimator_name(DS))

# print('estimators', estimators)
# print('='*100)
# print('classifiers', classifiers)

# scores_df = pd.DataFrame(scores)
# print(scores_df)


# #import joblib
# #save model as diabetes.pkl
# joblib.dump(DS,'ds.pkl')
# #loading the model
# model1 = joblib.load('ds.pkl')
