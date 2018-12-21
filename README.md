# Textile defect detection experiments

This repo contains an unorganized number of files used as experiments to classify the textile defects found in the [TILDA dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html).

The following approaches have been tried:

### 1. Anomaly (Outlier) detection

Based on an assumption that each feature's histogram can be approximated to a normal distribution. Well, the assumption was wrong.

### 2. Anomaly detection based on OneClassSVM

A class found in the scikit-learn library.

### 3. Simple Convolutional Neural Network.

### 4. Histogram of Oriented Gradients.

### 5. Siamese Network.

The problem with typical CNNs was that we had a very small dataset for training. Siamese networks on one hand are trained on a pair or triple of images. So, I had the idea of building pairs of images from 300 images (the size of the dataset used - a subset of the TILDA dataset) which would result in a much larger number of samples. Well, this worked to some point, but the complexity of the input images couldn't be distinguished successfully using this trick only.

### 6. Gabor filters.

Based on [Ajay Kumar's paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320303000050), gabor filters with different scales and orientations were used and tested on some samples. However, input images come in different orientations and scales which makes the use of gabor filters ineffective.

### 7. Complex Convolutional Neural Networks.

Experimented with 7 different models and architectures of CNNs to pick the best model and start making small modifications to it so that I can achieve a satisfying results. Although this was partially a success (training accuracy hit 97%), validation accuracy was about 62% and test accuracy didn't pass 50%. Part of the problem in my opinion is the bad, unfiltered and unprocessed dataset of TILDA which has ambiguous and noisy images that a human can mis-classify.
