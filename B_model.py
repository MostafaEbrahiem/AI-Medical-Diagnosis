import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.utils import to_categorical
from pickle import dump


def pure_cnn_model(X_train,y_train,SIZE):
    # Opening the files about data

    activation = 'relu'

    feature_extractor = Sequential()
    feature_extractor.add(Conv2D(32, 5, activation=activation, padding='same', input_shape=(SIZE, SIZE, 3)))
    feature_extractor.add(BatchNormalization())

    feature_extractor.add(Conv2D(32, 5, activation=activation, padding='same', kernel_initializer='he_uniform'))
    feature_extractor.add(BatchNormalization())
    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer='he_uniform'))
    feature_extractor.add(BatchNormalization())

    feature_extractor.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer='he_uniform'))
    feature_extractor.add(BatchNormalization())
    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Flatten())

    # Add layers for deep learning prediction
    x = feature_extractor.output
    x = Dense(128, activation=activation, kernel_initializer='he_uniform')(x)
    prediction_layer = Dense(6, activation='softmax')(x)

    # Make a new model combining both feature extractor and x
    cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
    cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(cnn_model.summary())
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    y_train1 = to_categorical(y_train1)
    y_test1 = to_categorical(y_test1)

    history = cnn_model.fit(X_train1, y_train1, epochs=5, validation_data=(X_test1, y_test1))

    dump(history, open('history.pkl', 'wb'))



    return history