# Architecture #1

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from image import ImageDataGenerator
import cv2

def image_preprocessing(img):
    return cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)

# Building the Convolutional Neural Network
###############################################################################
classifier = Sequential()
# Convolution Layer #1
classifier.add(Conv2D(16,5,strides=2,
                      padding='valid',
                      input_shape=(128,128,3),
                      activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Convolution Layer #2
classifier.add(Conv2D(32,3,
                      padding='same',
                      activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Fully Connected Layer
classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=6, activation='softmax'))
###############################################################################
# Compiling the network
classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
###############################################################################
# Preparing the image generator
training_set_path = "/textileconvnets/training_set/"
dev_set_path = "/textileconvnets/dev_set"
#test_set_path = "/home/osama/Documents/datasets/textile/dataset/test_set"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   preprocessing_function= image_preprocessing)

dev_datagen = ImageDataGenerator(rescale=1./255,
                                 preprocessing_function = image_preprocessing)
#test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        training_set_path,
        target_size=(128, 128),
        batch_size=27)

dev_set = dev_datagen.flow_from_directory(
        dev_set_path,
        target_size=(128, 128),
        batch_size=6)

#test_set = test_datagen.flow_from_directory(
#        '',
#        target_size=(128, 128),
#        batch_size=32,
#        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=40,
        epochs=50,
        validation_data=dev_set,
        validation_steps=60)





















