import numpy.random as rng
from keras import backend as K
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam


def euclid_dist(v):
    return ((v[0] - v[1]) ** 2) ** 0.5


def out_shape(shapes):
    return shapes[0]


class SiameseModel:
    def __init__(self):
        pass

    def build(self):
        input_shape = (150, 150, 3)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        convnet = Sequential()
        convnet.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(32, (3, 3), activation='relu'))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(64, (3, 3), activation='relu'))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(64, (3, 3), activation='relu'))
        convnet.add(Flatten())
        convnet.add(Dense(10, activation='sigmoid'))

        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)

        L1_distance = lambda x: K.abs((x[0] - x[1]))
        # both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        both = Lambda(euclid_dist, output_shape=out_shape)([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(both)
        siamese_net = Model(input=[left_input, right_input], outputs=prediction)
        return siamese_net
