from keras.layers import Activation, Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential


class CnnModel:

    def __init__(self):
        pass

    def build(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3)))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # model.add(Dense(2048))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(45))
        model.add(Activation('softmax'))
        return model


