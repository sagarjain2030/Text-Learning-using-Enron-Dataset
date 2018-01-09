#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#Importing GaussianNB
from sklearn.naive_bayes import GaussianNB

#Creating Classifier
clf = GaussianNB()

#Finding time required to fit classifier
t0 = time()

#Fit the classifier
clf.fit(features_train,labels_train)

#Calculating time required by classifier to Fit
print "training time:", round(time()-t0, 3), "s"

#Finding time required to predict
tp = time()

#Accuracy of classifier
print("The accuracy of classifier is ")
print(clf.score(features_test,labels_test))

#Calculating time required by classifier to Predict
print "testing time:", round(time()-tp, 3), "s"
#########################################################


