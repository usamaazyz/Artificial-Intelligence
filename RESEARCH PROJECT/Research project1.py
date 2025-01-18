#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import libraries
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import math
import pickle
import random
import operator
from collections import defaultdict

# Define a function to calculate the distance between two instances
def distance(instance1, instance2, k):
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

# Define a function to get the k nearest neighbors for a given instance
def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance,trainingset[x],k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [x[0] for x in distances[:k]]
    return neighbors

# Define a function to identify the class of a given instance based on its k nearest neighbors
def nearestclass(neighbors):
    classVote = defaultdict(int)
    for response in neighbors:
        classVote[response] += 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

# Define a function to evaluate the accuracy of a model
def getAccuracy(testSet, predictions):
    correct = sum(1 for i in range(len(testSet)) if testSet[i][-1] == predictions[i])
    return 1.0 * correct / len(testSet)

# Load the dataset
def loadDataset(filename, split):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    random.shuffle(dataset)
    train_size = int(split * len(dataset))
    trainingSet = dataset[:train_size]
    testSet = dataset[train_size:]
    return trainingSet, testSet

# Load the dataset and split it into training and test sets
trainingSet, testSet = loadDataset('my.dat', 0.7)

# Get the list of class labels
class_labels = defaultdict(str)
for i, folder in enumerate(os.listdir('Data/genres_original')):
    class_labels[i] = folder

# Classify the test instances and calculate the accuracy
predictions = [nearestclass(getNeighbors(trainingSet, test_instance, 5)) for test_instance in testSet]
accuracy = getAccuracy(testSet, predictions)
print(f"Accuracy: {accuracy}")

# Example usage for a new instance
file ='/Users/Usama Aziz/Downloads/RESEARCH/metal.00003.wav'
rate, signal = wav.read(file)
mfcc_feat = mfcc(signal, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(axis=0)
feature = (mean_matrix, covariance, None)
prediction = nearestclass(getNeighbors(trainingSet, feature, 5))
print(f"Prediction: {class_labels[prediction]}")


# In[ ]:




