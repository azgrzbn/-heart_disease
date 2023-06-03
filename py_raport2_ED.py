#CART

import pandas as pd
import numpy as np

heart = pd.read_csv('C:\\Users\\dell\\Downloads\\heart_disease.csv') 
korelacje = heart.corr()
korelacje

X_names = ["sex","age","chest_pain_type","resting_blood_pressure","serum_cholestoral","fasting_blood_sugar","resting_elect","max_heart_rate","angina","slope","vessel","thalassemia"]
X = heart[X_names]
y = heart[['heart_disease']]
print('Etykiety klas dla zmiennej celu:', np.unique(y)) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=308294, stratify=y)#póki co tylko mój indeks

from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=10,min_samples_leaf=5) 
tree1train = model1.fit(X_train,y_train)

y_names=["Nie","Tak"]
y_names

'''import graphviz |
from sklearn.tree import export_graphviz
wykres_drzewa1=export_graphviz(tree1train,out_file=None,filled=True,feature_names=X_names,class_names=y_names)
graph = graphviz.Source(wykres_drzewa1,format='png')
graph'''

from sklearn.metrics import confusion_matrix
def ocen_model_klasyfikacji_binarnej(y_true, y_pred, digits = 3):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(fn+tp)
    specificity = tn/(tn+fp)
    print('Trafność: ', round(accuracy, digits))
    print('Czułość: ', round(sensitivity, digits))
    print('Specyficzność: ', round(specificity, digits))
    
y_train_pred = model1.predict(X_train)  
y_score_train=model1.predict_proba(X_train)[:,1]

print('Zbiór uczący')
ocen_model_klasyfikacji_binarnej(y_train, y_train_pred)

y_test_pred = model1.predict(X_test)  
y_score_test=model1.predict_proba(X_test)[:,1]

print('Zbiór testowy')
ocen_model_klasyfikacji_binarnej(y_test, y_test_pred)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def ROC(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specyficzność')
    plt.ylabel('Czułość')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    
ROC(y_train, y_score_train)
ROC(y_test, y_score_test)

model1.feature_importances_

waznosci = pd.Series(model1.feature_importances_, index=X_names)
waznosci.sort_values(inplace=True)
waznosci.plot(kind='barh')