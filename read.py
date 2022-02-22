import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics,model_selection,preprocessing
import joblib


data=pd.read_csv('diabets.csv')
data.head()
#describe a DataFream and info
data.info()
data.describe()


#Convert text values to numerical values
data['Gender']=np.where(data['Gender']=='F',0,1)
data['CLASS']=np.where(data['CLASS']=='Y',0,1)



x=data.iloc[:,2:13] #Split the data  inputs(features (x))
print('feauter are : ','\n' ,x)

y=data.iloc[:,13:]#split the data output(target(y))
print('target is:','\n' ,y)

print(data)

#defining estimators function with a array to save the models after learning for bars chart ploting 
def print_estimator_name(estimator):
    return estimator.__class__.__name__

estimators=[] #estimators array
classifiers = [] #classifiers array to save classification models for confusion matrices ploting
scores = [] #scores array for saveing presicion and recall results for comparing
acc=[] #saving accuracy scores for bars chart ploting
f1=[] #saveing f1 score for bars chart ploting

x_normalizer=preprocessing.StandardScaler()
x_norm=x_normalizer.fit_transform(x)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x_norm,y,test_size=0.33,random_state=42)
print('x_train :',x_train.shape), print('y_train: ',y_train.shape)
print('x_test :',x_test.shape), print('y_test :',y_test.shape)
DS= DecisionTreeClassifier()
DS.fit(x_train, y_train)

y_pred = DS.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred,average='weighted')
recall = metrics.recall_score(y_test, y_pred,average='weighted')
f1_score = metrics.f1_score(y_test, y_pred,average='weighted')

print('accuracy',accuracy*100)
print('precision',precision*100)
print('recall',recall*100)
print('f1_score',f1_score*100)

classifiers.append(DS)
scores_dict = { "precision": precision, "recall": recall}
scores.append(scores_dict)
acc.append(accuracy)
f1.append(f1_score)
estimators.append(print_estimator_name(DS))

print('estimators', estimators)
print('='*100)
print('classifiers', classifiers)

scores_df = pd.DataFrame(scores)
print(scores_df)


#import joblib
#save model as diabetes.pkl
joblib.dump(DS,'diabetes.pkl') 
#loading the model
model1 = joblib.load('diabetes.pkl')
