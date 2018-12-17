#!/usr/bin/env python3
# from anomaly import Anomaly
from sklearn import svm
import numpy as np
# from math import exp
from os import listdir
from scipy import misc
import matplotlib.pyplot as plt
def loadImage(image_name):
    """
    image_name: the full path name of the image (including the image's name).
    Returns a flattened image of shape (1, Width * Heigh)
    """
    image = misc.imread(image_name)
    print("Reading image:", image_name)
    return image.reshape((1,-1))


def load_images(path):
    """
    path: the location of the training examples
    Returns a (M, N) matrix of all images in this location where:
        M is the number of examples/images.
        N is the number of features/pixels in each image.
    """
    file_names = listdir(path)
    X = []
    for file in file_names:
        X.append(loadImage(path + '//' + file))
    return np.asarray(X).reshape((len(file_names), -1))


# The code starts HERE
training_path = "training/positive"
X = load_images(training_path)
test_path = "training/positive_test"
X_test = load_images(test_path)
#detector = Anomaly(epsilon=0.0001)
#detector.fit(X)
min_cost = 1000
min_nu = 0
for n in np.linspace(0.1, 1, num=50):
    print("n =", n)
    clf = svm.OneClassSVM(nu=n, kernel='rbf', gamma=0.0001)
    clf.fit(X)
#    example = X[10:20]
    prediction = clf.predict(X)
    print(prediction)
    cost = np.sum(1 - prediction)
    print("cost =", cost)
    if cost < min_cost:
        min_cost = cost
        min_nu = float(n)
        print("Minimum mu =", min_nu)


# min_nu = 0.325
clf = svm.OneClassSVM(nu=min_nu, kernel='rbf', gamma=0.0001)
clf.fit(X)
examples = X_test[:]
prediction = clf.predict(examples)
print(prediction)
width, height = 768, 512
for i in range(len(examples)):
    plt.subplot(5, 4, i+1)
    plt.axis('off')
    plt.imshow(examples[i].reshape((height, width)))
    plt.title(str(prediction[i]))

plt.show()













