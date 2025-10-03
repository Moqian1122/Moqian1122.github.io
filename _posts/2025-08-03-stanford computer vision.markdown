---
layout: post
title:  "Learning with Stanford 01: deep learning for computer vision"
date:   2025-08-03 23:09:01 +0200
categories: Machine Learning, Computer Vision
---

This is a starting point of a series that I have been always thinking to initialize. Recently I got the energy and time to commence it. The series is about learning with stanford CS courses. Stanford CS courses are open-source and popular. I would like to record my learning journey with them on my personal website, as it would be a great asset for me. In the future, I hope my records could also help others who share a common situation with me. As a professional-to-be jumping in from literature studies, I image that other similar learners will have problems in understanding certain knowledge involved during these courses that those pure science guys will never foresee and explain. Hopefully they can expect a solution here.

I start with [CS231n: Deep Learning for Computer Vision (Spring 2025)](https://cs231n.stanford.edu/) as Episode 01 for the whole 'Learning with Stanford' series. While I follow the latest assignments, the videos available online are from previous semesters.

## Assignment 1: Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets

### A kNN Classifier

k-Nearest-Neighbor (kNN) is a clustering machine learning method. Compared to other classifiers such as linear classifiers and support vector machines which really train models to fit on training data, a kNN is not really training a model but simply remembering all training data. Once a new data point enters, the kNN will compare this single point to all other training data points based on a specific distance metric. Therfore, kNN is 'data as model'. That is to say, a nKK model is merely the remembered training data itself. k is a major hyperparameter in a kNN classifier. k could be chosen via cross-validation (CV).

Based on the idea, we want to implement with codes. Following the spirits of object-oriented-programming (OOP), each kind of machine learning models should be a class. For a specific task, an 'object' should be initialized from the class. Of course, some methods should be attached to the object to function as training, hyperparameter tuning and prediction. Nowadays, universities are very generous to give students 'starter codes'. However, since we aim at landing on positions like machine learning engineers, we would try to build it from scratch.

We start our code implementation from here. First of all, we import 'object' from 'builtins'. This is a heritage from Python 2. In Python 3 we do not have to do so. For the sake of compatibility, we would like to keep this old style. When initializing a kNN object, we do not really need to do something, so we simply give it a 'pass'. Three key methods should be training and prediction. To make predictions, comparision between the new data points and the training data points based on a certain similarity metric is necessary. Therefore, extra similarity measurement functions are added to an object. As mentioned above, kNN simply 'remembers' the training data. So, value assignment operation is enough in the training process. I completed the starter code. However, here I post the code that I wrote from scratch while refering to the starter code in terms of the structure.

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
        dists = np.zeros((X.shape[0], self.X_train.shape[0]))
        dists = np.absolute(np.sum(X, axis=1)[:,np.newaxis] - np.sum(self.X_train, axis=1)[:,np.newaxis].T)
        return dists

    def compute_distance_l2(self, X):
        dists = np.zeros((X.shape[0], self.X_train.shape[0]))
        dists = np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(self.X_train**2, axis=1) - 2*np.dot(X, self.X_train.T))
        return dists

    def predict_labels(self, dists, k):
        y_pred = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            cloest_y = []
            cloest_y_indices = np.argsort(dists[i], kind='heapsort')[:k]
            labels, counts = np.unique(self.y_train[cloest_y_indices], return_counts=True)
            cloest_y = np.asarray((labels, counts)).T
            y_pred[i] = np.min(cloest_y[cloest_y[:,1]==np.max(cloest_y[:,1])][:,0])
        return y_pred

    def predict(self, X, metric, k=1):
        if metric == 'l1':
            dists = self.compute_distance_l1(X)
        elif metric == 'l2':
            dists = self.compute_distance_l2(X)
        else:
            raise ValueError("Invalid value entered. Please enter either 'l1' or 'l2'")
        return self.predict_labels(dists, k=k)
```

The computation of the L1 and L2 distances is the hardest part for me. In the starter code, three solutions regarding to different loops (2 loops, 1 loop and 0 loops) are provided. Obviously, the 0-loop solution is the fastest one. I struggled a bit with the vector quantization. The most import thing is to utilize broadcasting mechanism of numpy and matrix transpose. While the arrays are one-dimensional, we could make use of 'np.newaxis' to add one dimension.

The L1 computation is slightly easier as long as we could realize:

$$
\sum^{P}|I^{p}_{a}-I^{p}_{b}| = |\sum^{P}I^{p}_{a}-\sum^{P}I^{p}_{b}|
$$

By doing so, we could sum the input X and X_train separately alongside axis=1 to yield two one-dimensional arrays. Then, with the np.newaxis we could use the broadcasting mechanisms of numpy to transform them to shapes (X.shape[0], 1) and (X_train.shape[0], 1) respectively. However, we can note that the two arrays are still not operatable. It's easy to find transposing any of them makes life easier for us. We will have:

```python
dists = np.absolute(np.sum(X, axis=1)[:,np.newaxis] - np.sum(self.X_train, axis=1)[:,np.newaxis].T)
```

The practice uses CIFAR-10 data set. 'CIFAR' represents Canadian Institute for Advanced Research. 10 refers to that the data set consist of 10 classes, represented by 0 to 9. It contains 60,000 images with 50,000 images as training data and 10,000 images as test data. Each image has 3,072 pixel values with the first 1,024 the red values, the middle 1,024 the green values and the last 1,024 the blue values. Each class has exactly 5,000 training images and exactly 1,000 test images.

![The preview of a subtle set of the pictures'](/assets/images/computer_vision_knn.png)

To get an optimal knn for this classification task, we use 5-fold cross-validation to tune the optimal k value as well as the selection of an optimal similarity measurement (interchangably used with 'distance computation'). The visualization of the accuracy against a set of k values for both L1 and L2 is demonstated as followed.

![The visualization of prediction accuracy against different k values for both L1 and L2'](/assets/images/cross_validation.png)

When metric is L1 and k is seleced as 38, the 'model' is optimal. We initialize a kNN classifier with this combination. Next, we need to prepare a mini test with only one instance. Preparing it helps us to undertand how to preprocess images.

```python
knn = KNearestNeighbor()
knn.train(X_train, y_train)
```

To really get familiar with the pipeline (after we have a kNN 'model' with optimal k value and similarity measurement), I randomly find a frog image online. It's nice to diverge from here so we could also take a look at OpenCV, a popular computer vision framework. The original  image is 750\*455. I already cropped it outside this notebook to 455\*455. To make it compatible with the kNN model, we use cv2.resize to make it to a shape of (32, 32). The original image and the resized image are displayed below. The resized image on the right-hand side has lower image quality. It makes sense since the image is downsized.

The goal is to finally have an image array of shape (1, 3072). To realize the goal, 4 key steps follow:\
(1) We use imread from OpenCV to read images as arrays. However, a trick is that it automatically transforms images as BGR instead of RGB. It is fine in general but might cause a mismatch issue here as CIFAR-10 data set is in the order of RGB pixelwise;\
(2) The good news is that we could use cv2.COLOR_BGR2RGB to easily switch from BGR to RGB. Now, the image array has a shape of (32, 32, 3);\
(3) We concatenate the red channel, the green channel and the blue channel respectively. After that, we will have a flattened one-dimensional array (3072, );\
(4) Reshaping it helps us to get the ideal shape of (1, 3072).

![The frog picture'](/assets/images/frog_image.png)

```python
import cv2

frog_resized = cv2.resize(cv2.imread(os.path.expanduser('~/learning/learn_cv/computer_vision_frog.jpg'), flags=-1), (32,32))
frog_resized_rgb = cv2.cvtColor(frog_resized, cv2.COLOR_BGR2RGB)
frog_resized_rgb_flattened = np.concatenate((np.concatenate(frog_resized_rgb[:,:,0]), np.concatenate(frog_resized_rgb[:,:,1]), np.concatenate(frog_resized_rgb[:,:,2])))
frog_resized_rgb_flattened = frog_resized_rgb_flattened.reshape(1,3072)
```

Now, we get the ideal flattened image array following the RGB code. Let's predict it! We provide the class list and the integer format of the label number we get will yield the class name corresponding to the label.

```python
frog_pred = knn.predict(frog_resized_rgb_flattened, metric='l1', k=38)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Tah-dah~ The image is classified as %s!" % classes[int(frog_pred[0])])
```

We get the outcome: 'Tah-dah~ The image is classified as frog!' It's nice to have the outcome. However, the accuracy is not high. We might want to keep improving it afterwards.

Besides the general pipeline of the exercise, there are two tricks to pay attention to:\
(1) When we work with images and videos, we have to operate with large numeric arrays. It is important to know faster vector quantization;\
(2) There are tricks to be noted when we preprocess (read, display or transform) images with frameworks such as OpenCV. It requires carefulness.