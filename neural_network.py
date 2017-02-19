import numpy as np
from os import listdir
from skimage.color import rgb2gray

import cv2
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from mapping import connect_name_and_number


def to_categorical(labels, n):
    ret_val = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    ret_val[ll[:, 0], ll[:, 1]] = 1
    return ret_val


def learn_nn(model):
    train = np.zeros((1, 900))
    out = np.array([])
    folders = listdir('dataset')
    for folder in folders:
        filenames = listdir('dataset/'+str(folder))
        for filename in filenames:
            train_image = cv2.imread('dataset/' + str(folder) + '/' + str(filename))
            train_image = rgb2gray(train_image)
            train_image = train_image.reshape(1, 900)
            train_image *= 255
            train_image = train_image[0]

            train = np.vstack([train, train_image])
            # train.append(train_image)
            out = np.append(out, connect_name_and_number(folder))
            # out.append(0)

    train_out = to_categorical(out.astype('int'), 14)
    train = np.delete(train, 0, axis=0)

    model.add(Dense(70, input_dim=900))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(14))
    model.add(Activation('tanh'))

    sgd = SGD(lr=0.1, decay=0.0001, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    training = model.fit(train, train_out, nb_epoch=5000, batch_size=400, verbose=1)
    print training.history['loss'][-1]

    return model
