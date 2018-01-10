#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
#Importing SVM
from sklearn import svm

#Creating Classifier
clf = svm.SVC(kernel= 'linear')
#Using Linear Function as kernel

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



#########################################################
#Training on small dataset
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#Creating Classifier
clf1 = svm.SVC(kernel= 'linear')
#Using Linear Function as kernel

#Finding time required to fit classifier
t0 = time()

#Fit the classifier
clf1.fit(features_train,labels_train)

#Calculating time required by classifier to Fit
print "training time:", round(time()-t0, 3), "s"

#Finding time required to predict
tp = time()

#Accuracy of classifier
print("The accuracy of classifier is ")
print(clf1.score(features_test,labels_test))

#Calculating time required by classifier to Predict
print "testing time:", round(time()-tp, 3), "s"
########################################################



#########################################################
#Training on small dataset using Radial basis function as kernel
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#Creating Classifier
clf2 = svm.SVC(kernel='rbf',C=10000)

#Finding time required to fit classifier
t0 = time()

#Fit the classifier
clf2.fit(features_train,labels_train)

#Calculating time required by classifier to Fit
print "training time:", round(time()-t0, 3), "s"

#Finding time required to predict
tp = time()

#Accuracy of classifier
print("The accuracy of classifier is ")
print(clf2.score(features_test,labels_test))

#Calculating time required by classifier to Predict
print "testing time:", round(time()-tp, 3), "s"
########################################################



#########################################################
#Training on small dataset using Radial basis function with C = 10000 as kernel
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#Creating Classifier
clf3 = svm.SVC(kernel='rbf',C=10000)

#Finding time required to fit classifier
t0 = time()

#Fit the classifier
clf3.fit(features_train,labels_train)

#Calculating time required by classifier to Fit
print "training time:", round(time()-t0, 3), "s"

#Finding time required to predict
tp = time()

#Accuracy of classifier
print("The accuracy of classifier is ")
print(clf3.score(features_test,labels_test))

#Calculating time required by classifier to Predict
print "testing time:", round(time()-tp, 3), "s"
########################################################
pred = clf.predict(features_test)

print '10 =  ',pred[10]
print '26 =  ',pred[26]
print '50 =  ',pred[50]

x = [s for s in pred if s==1]
print(len(x))