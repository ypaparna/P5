#!/usr/bin/python

####import the necessary modules
import pickle
import math
import sys
import matplotlib.pyplot as plt
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


### Task 1: Select what features you'll use.
###  Before selecting features load the data into pandas dataframe and explore

target_label = ['poi']
email_features_list = [
'shared_receipt_with_poi'
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

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#plot some of the data points to see outliers (salary vs bonus)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
    

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Graphical view of from and to emails
features = ["from_messages", "to_messages"]
data = featureFormat(data_dict, features)
for point in data:
    from_messages = point[0]
    to_messages = point[1]
    plt.scatter( from_messages, to_messages )
    

plt.xlabel("from_messages")
plt.ylabel("to_messages")
plt.show()

### Graphical view of from_this_person_to_poi and to from_poi_to_this_person
features = ["from_this_person_to_poi", "from_poi_to_this_person"]
data = featureFormat(data_dict, features)
for point in data:
    from_this_person_to_poi = point[0]
    from_poi_to_this_person = point[1]
    plt.scatter( from_this_person_to_poi, from_poi_to_this_person )
    

plt.xlabel("from_this_person_to_poi")
plt.ylabel("from_poi_to_this_person")
plt.show()

### Remove outliers from the dictionary 
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E",0)

### Task 3: Create new feature(s)
##Add new features
#Chose two new features - ratio of messages received from poi and ratio of messages to poi
for employee in data_dict:
    data_dict[employee]['ratio_of_messages_received_from_poi'] = float(data_dict[employee]['from_poi_to_this_person'])/float(data_dict[employee]['to_messages'])
    if math.isnan(float(data_dict[employee]['ratio_of_messages_received_from_poi'])):
          data_dict[employee]['ratio_of_messages_received_from_poi'] = 0
    
    data_dict[employee]['ratio_of_messages_sent_to_poi'] = float(data_dict[employee]['from_this_person_to_poi'])/float(data_dict[employee]['from_messages'])
    if math.isnan(float(data_dict[employee]['ratio_of_messages_sent_to_poi'])):
        data_dict[employee]['ratio_of_messages_sent_to_poi'] = 0

new_features_list = [
'ratio_of_messages_received_from_poi',
'ratio_of_messages_sent_to_poi'
]

##Add new features to the features list
my_features_list = my_features_list + new_features_list
print "my feature list:", my_features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##split data into training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# Start initial analysis
#### Try Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking for all the initial features selected: '
for i in range(11):
    print "feature no. {}: {} ({})".format(i+1,my_features_list[indices[i]+1],importances[indices[i]])

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



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

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

###Let us also try DecisionTree classifier and kbest to select features and tune parameters 
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
print "feature importances for the features selected by kbest ", importances
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})".format(i+1,features_selected[indices[i]],importances[indices[i]])

##use this cassifier for validation
test_classifier(clf, my_dataset, my_features_list)

#Decision tree classifier gave the best accuracy and precision

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)