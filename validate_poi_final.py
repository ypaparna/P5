#!/usr/bin/python


"""
    Identify Enron Employees who may have committed fraud based on the public
    Enron financial and email dataset using Machine Learning
"""
    
####import the necessary modules
import pickle
import math
import sys
import matplotlib
import pandas
from time import time
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tester import test_classifier
from tester import dump_classifier_and_data
sys.path.append("../tools/")

##Start by loading data into a dictionary

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

##Add new features
#Chose two new features - ratio of messages received from poi , ration of messages to poi
for employee in data_dict:
    data_dict[employee]['ratio_of_messages_received_from_poi'] = float(data_dict[employee]['from_poi_to_this_person'])/float(data_dict[employee]['to_messages'])
    if math.isnan(float(data_dict[employee]['ratio_of_messages_received_from_poi'])):
          data_dict[employee]['ratio_of_messages_received_from_poi'] = 0
    
    data_dict[employee]['ratio_of_messages_sent_to_poi'] = float(data_dict[employee]['from_this_person_to_poi'])/float(data_dict[employee]['from_messages'])
    if math.isnan(float(data_dict[employee]['ratio_of_messages_sent_to_poi'])):
        data_dict[employee]['ratio_of_messages_sent_to_poi'] = 0
 

#plot some of the data points to see outliers (salary vs bonus)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
    

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Graphical view of from and to emails
features = ["from_messages", "to_messages"]
data = featureFormat(data_dict, features)
for point in data:
    from_messages = point[0]
    to_messages = point[1]
    matplotlib.pyplot.scatter( from_messages, to_messages )
    

matplotlib.pyplot.xlabel("from_messages")
matplotlib.pyplot.ylabel("to_messages")
matplotlib.pyplot.show()

### Graphical view of from_this_person_to_poi and to from_poi_to_this_person
features = ["from_this_person_to_poi", "from_poi_to_this_person"]
data = featureFormat(data_dict, features)
for point in data:
    from_this_person_to_poi = point[0]
    from_poi_to_this_person = point[1]
    matplotlib.pyplot.scatter( from_this_person_to_poi, from_poi_to_this_person )
    

matplotlib.pyplot.xlabel("from_this_person_to_poi")
matplotlib.pyplot.ylabel("from_poi_to_this_person")
matplotlib.pyplot.show()

##For easy analysis load the dictionary into a Pandas dataframe
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)

###Explore data
print "What is the Size of the dataset?", df.shape
print "Count of POIs in the dataset", len(df[(df['poi']>0)])
print "Count of non-POIs in the dataset", len(df[(df['poi']==0)])
print "No. of records with missing total_payments ", len(df[(df['total_payments']=='NaN')])
print "No. of records with missing salary ", len(df[(df['salary']=='NaN')])
print "No. of records with missing bonus ", len(df[(df['bonus']=='NaN')])
print "No. of records with missing director_fees", len(df[(df['director_fees']=='NaN')])
print "No. of records with missing restricted_stock_deferred", len(df[(df['restricted_stock_deferred']=='NaN')])
print "No. of records with missing deferral_payments", len(df[(df['deferral_payments']=='NaN')])
print "No. of records with missing from_messages", len(df[(df['from_messages']=='NaN')])
print "No. of records with missing from_poi_to_this_person", len(df[(df['from_poi_to_this_person']=='NaN')])
print "No. of records with missing from_this_person_to_poi", len(df[(df['from_this_person_to_poi']=='NaN')])
print "No. of records with missing shared_receipt_with_poi", len(df[(df['shared_receipt_with_poi']=='NaN')])
print "No. of records with missing to_messages", len(df[(df['to_messages']=='NaN')])
print "No. of records with missing deferred_income", len(df[(df['deferred_income']=='NaN')])
print "No. of records with missing exercised_stock_options", len(df[(df['exercised_stock_options']=='NaN')])
print "No. of records with missing expenses", len(df[(df['expenses']=='NaN')])
print "No. of records with missing loan_advances", len(df[(df['loan_advances']=='NaN')])
print "No. of records with missing long_term_incentive", len(df[(df['long_term_incentive']=='NaN')])
print "No. of records with missing restricted_stock", len(df[(df['restricted_stock']=='NaN')])
print "No. of records with missing total_stock_value", len(df[(df['total_stock_value']=='NaN')])
print "No. of records with missing other", len(df[(df['other']=='NaN')])



# set the index of df to be the employees series:

df.replace('NaN', 0, inplace = True)

# identify the bonus outlier seen in the plot 
print df["bonus"].argmax()
##Identify the outlier 
df.drop('TOTAL')


### Remove outliers from the dictionary 
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


##Select all the features including the new features for the initial try
# first element is the label, and the rest are features

target_label = ['poi']
email_features_list = [
'shared_receipt_with_poi',
'ratio_of_messages_received_from_poi',
'ratio_of_messages_sent_to_poi'
]
financial_features_list = [
'bonus',
'exercised_stock_options',
'expenses',
'long_term_incentive',
'restricted_stock',
'salary',
'total_payments',
'total_stock_value',
]
my_features_list = target_label + financial_features_list + email_features_list

##load data my_dataset 
my_dataset = data_dict


####Use all features in the initial analysis

data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##split data into training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# Start initial analysis
#### Try Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### calculate and return the accuracy on the test data    
acc = accuracy_score(pred, labels_test)
print "Decision Tree accuracy: ", acc

##Calculate precision and recall scores
True_POIs=sum(float(num) == 1.0 for num in labels_test)
print "No. of POIs in the test set", True_POIs
## What is the test set size?
test_set_size = len(labels_test)
print "Total no. of people in the test set", test_set_size
##Identify True positives

count =0
for i in range(len(labels_test)):
    if labels_test[i] ==1.0:
        if labels_test[i] == pred[i]:        
            count = count+1

print "True Positives", count
print "precision score using initial analysis", precision_score(labels_test, pred, average='binary')
print "recall score using initial analysis", recall_score(labels_test, pred, average='binary')

##Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
t0 = time()
clf = clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
########predictions
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

 ### calculate and return the accuracy on the test data
acc = accuracy_score(pred, labels_test)
print "Naive Bayes accuracy: ", acc
print "NB precision score", precision_score(labels_test, pred, average='micro')
print "NB recall score", recall_score(labels_test, pred, average='micro')

######### SVM Classifier#########
clf = SVC()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t1, 3), "s"

acc = accuracy_score(pred, labels_test)
print "svm accuracy: ", acc
print "SVM precision score", precision_score(labels_test, pred, average='micro')
print "SVM recall score", recall_score(labels_test, pred, average='micro')


##Explore NaiveBayes further with feature selection and parameter tuning

print "Feature Selection and parameter tuning for NB\n"
scaler = MinMaxScaler()
kbest = SelectKBest(f_classif)
clf_NB = GaussianNB()
pipeline =  Pipeline(steps=[('scaling', scaler),("kbest", kbest), ("NB", clf_NB)])
parameters = {'kbest__k': [1,2,3,4,5,6,7]}
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
gs = GridSearchCV(pipeline, parameters, scoring='f1', cv=cv)
gs.fit(features, labels)

clf = gs.best_estimator_
print "NaiveBayes", clf 

# Access the SelectKBest features selected
features_selected=[my_features_list[i+1] for i in clf.named_steps['kbest'].get_support(indices=True)]
print "features selected by kbest using NB", features_selected

####Validate the above classfier using test_classifier 
test_classifier(clf, my_dataset, my_features_list)

###Let us try DecisionTree classifier and kbest 
print "Feature Selection and parameter tuning for Decision Tree\n"
scaler = MinMaxScaler()
kbest = SelectKBest(f_classif)
clf_DTC = DecisionTreeClassifier(random_state=42)
pipeline =  Pipeline(steps=[('scaling', scaler),("kbest", kbest), ("DTC", clf_DTC)])
parameters = {'kbest__k': [1,2,3,4,5,6,7],
            'DTC__criterion': ['gini', 'entropy'],
            'DTC__min_samples_split': [2, 10, 20],
            'DTC__max_depth': [None, 2, 5, 10],
            'DTC__min_samples_leaf': [1, 5, 10],
            'DTC__max_leaf_nodes': [None, 5, 10, 20]}
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
gs = GridSearchCV(pipeline, parameters, scoring='f1', cv=cv)
gs.fit(features, labels)

clf = gs.best_estimator_
print "decision tree", clf 

# Access the SelectKBest features selected
features_selected=[my_features_list[i+1] for i in clf.named_steps['kbest'].get_support(indices=True)]
print "features selected by kbest", features_selected

# Access the feature importances
importances = clf.named_steps['DTC'].feature_importances_
print "feature importances ", importances
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])

##use this cassifier for validation
test_classifier(clf, my_dataset, my_features_list)
###DecisionTree classifier tuning gave the best scores in the analysis
###Dump DecisioTree classifier and data for testing
dump_classifier_and_data(clf, my_dataset, my_features_list)
#####################end it here
