---
layout: post
title:  "Learning with Stanford 01: deep learning in computer vision"
date:   2025-08-03 23:09:01 +0200
categories: Machine Learning, Computer Vision
---

This is a starting point of a series that I have been always thinking to initialize. Recently I got the energy and time to commence it. The series is about learning with stanford CS courses. Stanford CS courses are open-source and popular. I would like to record my learning journey with them on my personal website, as it would be a great asset for me. In the future, I hope my records could also help others who share a common situation with me. As a professional-to-be jumping in from literature studies, I image that other similar learners will have problems in understanding certain knowledge involved during these courses that those pure science guys will never foresee and explain. Hopefully they can expect a solution here.

I start with [CS231n: Deep Learning for Computer Vision (Spring 2025)](https://cs231n.stanford.edu/) as Episode 01 for the whole 'Learning with Stanford' series. While I follow the latest assignments, the videos available online are from previous semesters.

## Assignment 1: Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets

### A kNN Classifier

k-Nearest-Neighbor (kNN) is a clustering machine learning method. Compared to other classifiers such as linear classifiers and support vector machines which really train models to fit on training data, a kNN is not really training a model but simply remembering all training data. Once a new data point enters, the kNN will compare this single point to all other training data points based on a specific distance metric. Therfore, kNN is 'data as model'. That is to say, a nKK model is merely the remembered training data itself. k is a major hyperparameter in a kNN classifier. k could be chosen via cross-validation (CV).

Based on the idea, we want to implement with codes. Following the spirits of object-oriented-programming (OOP), each kind of machine learning models should be a class. For a specific task, an 'object' should be initialized from the class. Of course, some methods should be attached to the object to function as training, hyperparameter tuning and prediction. Nowadays, universities are very generous to give students 'starter codes'. However, since we aim at landing on positions like machine learning engineers, we would try to build it from scratch.

We start our code implementation from here. First of all, we import 'object' from 'builtins'. This is a heritage from Python 2. In Python 3 we do not have to do so. For the sake of compatibility, we would like to keep this old style. When initializing a kNN object, we do not really need to do something, so we simply give it a 'pass'. Three key methods should be training and prediction. To make predictions, comparision between the new data points and the training data points based on a certain similarity metric is necessary. Therefore, extra similarity measurement functions are added to an object. As mentioned above, kNN simply 'remembers' the training data. So, value assignment operation is enough in the training process.

```python
from builtins import object
import numpy as np

class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distance_l1(self, X):
        pass

    def compute_distance_l2(self, X):
        pass

    def predict_labels(self, dists, k=1):
        pass

    def predict(self, X, k=1, metric='l2'):
        pass
```