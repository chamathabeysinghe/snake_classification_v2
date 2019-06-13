from keras.layers import Activation, Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential


class ResnetModel:

    def __init__(self):
        pass

    def build(self):
        model = Sequential()
        model.add(Dense(1536, activation='relu', input_shape=(1536,)))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(45))
        model.add(Activation('softmax'))
        return model


