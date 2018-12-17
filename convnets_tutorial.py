# Convolution Neural Network

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(16,
                             3,
                             3,
                             border_mode = 'valid',
                             input_shape = (128, 128, 1),
                             activation = 'relu'))
# Step 2 - Max pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'softmax'))

# Step 5 - Compiling
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

dev_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/home/osama/Documents/datasets/textile/dataset/training_set/',
        target_size=(128, 128),
        batch_size=1)

dev_set = dev_datagen.flow_from_directory(
        '/home/osama/Documents/datasets/textile/dataset/dev_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/home/osama/Documents/datasets/textile/dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

import matplotlib.pyplot as plt
import numpy as np
plt.figure(2)
for i, im in enumerate(training_set):
    plt.xticks([])
    plt.yticks([])
    plt.title(np.argwhere(im[1][0,:] > 0) + 1)
#    print(im[0].shape)
    plt.imshow(im[0][0,:,:,0], cmap='gray')
    if i > 3:
        break
    plt.subplot(2,2,i+1)

classifier.fit_generator(
        training_set,
        steps_per_epoch=1080,
        epochs=25,
        validation_data=dev_set,
        validation_steps=360)