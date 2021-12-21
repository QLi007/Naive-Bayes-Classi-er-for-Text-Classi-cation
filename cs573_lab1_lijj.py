# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:48:51 2020

@author: lijj
"""
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
#########################
#import data
with open ('test_data.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    rows=[row for row in reader]
testdata=np.array(rows)

with open ('test_label.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    rows=[row for row in reader]
testlabel=np.array(rows)
print(testdata[2,:2])

with open ('train_data.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    rows=[row for row in reader]
traindata=np.array(rows)
print(traindata[2,:3])

with open ('train_label.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    rows=[row for row in reader]
trainlabel=np.array(rows)
print(trainlabel[2,:3])

with open ('map.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    rows=[row for row in reader]
map=np.array(rows)
print(map[2,:1])

with open('vocabulary.txt','r') as txtfile:
    reader =csv.reader(txtfile)
    rows=[row for row in reader]
vocabulate=np.array(rows)
print(vocabulate[2])

#################prior
trainDocSize = int(traindata[traindata.shape[0] - 1, 0])
testDocSize = int(testdata[testdata.shape[0] - 1, 0])

prior_w=np.zeros(21)
#print (prior_w)
for i in range(0,trainlabel.shape[0],1):
    k=int(trainlabel[i])
    prior_w[k]=prior_w[k]+1
prior_w=prior_w[1:]
prior=(prior_w/prior_w.sum())
print(prior_w)

print(prior)
#######################word number
N=vocabulate.size
print('vocabulate.size is:', N)
#######################total work  number in different categorys
totalnum=np.zeros(prior.size).reshape(-1,1)
temp=np.zeros(traindata.shape[0]).reshape(-1,1)
temptrain=np.hstack((traindata,temp))

for i in range(traindata.shape[0]):
    k2=int(traindata[i,0])
    temptrain[i,3]=trainlabel[k2-1,0]#trainlabel start from 0
#print(temptrain[:72980,:])

for i in range(traindata.shape[0]):
#for i in range(100):
    k3=int(temptrain[i,3])
    #print (k3)
    totalnum[k3-1,0]=int(totalnum[k3-1,0])+int(temptrain[i,2])
print(totalnum.reshape(1,-1))
#####################################
##Calculate nk: number of times word wk occurs in all documents in class !j .
wordcount=np.zeros((N,prior.size))
for i in range(traindata.shape[0]):
    wj=int(temptrain[i,3])-1 
    vol_index=int(temptrain[i,1])-1 #index start from 1 in the csv
    vol_count=int(temptrain[i,2])
    wordcount[vol_index,wj]=int(wordcount[vol_index,wj])+vol_count
#print(wordcount[:10,:5])

######################
#Calcualte Maximum Likelihood estimator PMLE(wkj!j)
p_mle=wordcount/(totalnum.reshape(1,-1))
#print(p_mle[:5,:5])
logp_mle=np.log(p_mle)
logp_mle=np.nan_to_num(logp_mle)
#print(p_mle[:5,:5])

#######BE
p_be=(wordcount+1)/(totalnum.reshape(1,-1)+N)
#print(p_be[:5,:5])
logp_be=np.log(p_be)#(61188, 20)
#print(p_be[:5,:5])
####################
#train data MLH
print('Performance on Training Data')
print(logp_be.shape)#(61188, 20)
   
traindatatemp = np.zeros((trainDocSize, N))
for i in range(traindata.shape[0]):
    doc = int(traindata[i, 0])
    voc = int(traindata[i, 1])
    traindatatemp[doc - 1, voc - 1] = int(traindatatemp[doc - 1, voc - 1]) + int(traindata[i, 2])
traindata = traindatatemp.T

testdatatemp = np.zeros((testDocSize, N))
for i in range(testdata.shape[0]):
    doc = int(testdata[i, 0])
    voc = int(testdata[i, 1])
    testdatatemp[doc - 1, voc - 1] = int(traindatatemp[doc - 1, voc - 1]) + int(testdata[i, 2])
testdata = testdatatemp.T
log_prior=np.log(prior).reshape(-1,1)

#print(traindata.shape) #(61188, 11269)
#print(testdata.shape) #(61188, 7505)
#print(log_prior.T) 

trainmaxlh=np.zeros((prior.size,trainDocSize))
trainmaxlh=trainmaxlh+log_prior
trainmaxlh=trainmaxlh+ np.dot(logp_be.T,traindata)
trainpredit_be = (np.argmax(trainmaxlh, axis=0) + 1).reshape(trainlabel.shape[0], 1).astype(int)
trainpredit_be = confusion_matrix(trainlabel.astype(int), trainpredit_be.astype(int))
trainpredit_be_diagonal = trainpredit_be.diagonal().reshape(prior_w.size, 1)
trainoverall_accurate = np.sum(trainpredit_be_diagonal) / trainDocSize

print('Overall Accuracy = ',trainoverall_accurate)
traingroup_accurate = trainpredit_be_diagonal / prior_w.reshape(-1,1) 

print(trainpredit_be_diagonal)
print(prior_w.reshape(-1,1))
#print(traingroup_accurate)
print('Class Accuracy:')
for i in range(traingroup_accurate.shape[0]):
    print('Group ' + str(i+1) + ': ' + str(traingroup_accurate[i, 0]))
print('Confusion Matrix:')
print(trainpredit_be)
############################ mle

trainmaxlh_mle = np.zeros((prior_w.size, trainDocSize))
trainmaxlh_mle = trainmaxlh_mle + log_prior
trainmaxlh_mle = trainmaxlh_mle + np.dot(logp_mle.T, traindata)
trainpredit_mle = (np.argmax(trainmaxlh_mle, axis=0) + 1).reshape(trainlabel.shape[0], 1).astype(int)
trainpredit_mle = confusion_matrix(trainlabel.astype(int), trainpredit_mle.astype(int))
trainpredit_mle_diagonal = trainpredit_mle.diagonal().reshape(prior_w.size, 1) 

trainoverallaccurate = np.sum(trainpredit_mle_diagonal) / trainDocSize 
print('Overall Accuracy = ',trainoverallaccurate)

traingroup_accurate =trainpredit_mle_diagonal / prior_w.reshape(-1,1) 
print('Class Accuracy:')
for i in range(traingroup_accurate.shape[0]):
    print('Group ' + str(i+1) + ': ' + str(traingroup_accurate[i, 0]))

print('Confusion Matrix:')
print(trainpredit_mle)

# Performance on Testing Data

print(' Performance on Testing Data')
#######be
unique, testnum = np.unique(testlabel, return_counts=True)
testnum = testnum.reshape(prior_w.size, 1)

testmaxlh = np.zeros((prior_w.size, testDocSize))
testmaxlh = testmaxlh + log_prior
testmaxlh = testmaxlh + np.dot(logp_be.T, testdata)
testpredit_be = (np.argmax(testmaxlh, axis=0) + 1).reshape(testlabel.shape[0], 1).astype(int)

testpredit_be = confusion_matrix(testlabel.astype(int), testpredit_be.astype(int))

testpredit_bediagonal = testpredit_be.diagonal().reshape(prior_w.size, 1) 

testoverallacc = np.sum(testpredit_bediagonal) / testDocSize 
print('Overall Accuracy = ',testoverallacc)

testgroupacc = testpredit_bediagonal / testnum 

print('Class Accuracy:')
for i in range(testgroupacc.shape[0]):
    print('Group ' + str(i+1) + ': ' + str(testgroupacc[i, 0]))

print('Confusion Matrix:')
print(testpredit_be)
###################################
# Performance on Testing Data MLE

unique, testnum = np.unique(testlabel, return_counts=True)
testnum = testnum.reshape(prior_w.size, 1)

testmaxlh_mle = np.zeros((prior_w.size, testDocSize))
testmaxlh_mle = testmaxlh_mle + log_prior
testmaxlh_mle = testmaxlh_mle + np.dot(logp_mle.T, testdata)
testpredit_mle = (np.argmax(testmaxlh_mle, axis=0) + 1).reshape(testlabel.shape[0], 1).astype(int)
testpredit_mle = confusion_matrix(testlabel.astype(int), testpredit_mle.astype(int))
testpredit_mlediagonal = testpredit_mle.diagonal().reshape(prior_w.size, 1) 

testoverallacc = np.sum(testpredit_mlediagonal) / testDocSize 
print('Overall Accuracy = ',testoverallacc)

testgroupacc = testpredit_mlediagonal / testnum
print('Class Accuracy:')
for i in range(testgroupacc.shape[0]):
    print('Group ' + str(i+1) + ': ' + str(testgroupacc[i, 0]))

print('Confusion Matrix:')
print(testpredit_mle)