# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:55:27 2019

@author: Sreenivasa Reddy
"""

#building cnn
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
#sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
#from keras import backend
#assert len(backend.tensorflow_backend._get_available_gpus()) > 0

from keras.layers import Dropout
classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(128,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(256,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))



classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=1, activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/Sreenivasa Reddy/Desktop/Deep_Learning_A_Z/Volume_1_Supervised_Deep_Learning/Part2_Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/Sreenivasa Reddy/Desktop/Deep_Learning_A_Z/Volume_1_Supervised_Deep_Learning/Part2_Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=35,
        validation_data=test_set,
        validation_steps=2000/32)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
