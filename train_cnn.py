from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from model.CNNModelComplex import CnnModel
from keras.optimizers import Adam
from time import time
import glob
from keras.models import load_model
from keras.utils import multi_gpu_model
from data.shared import base_directory

batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1./255.0
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_directory, 'train'),
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_directory, 'val'),
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

os.makedirs("checkpoints", exist_ok=True)
file_path = "checkpoints/cnn-model-epoch-{epoch:05d}-train_loss-{loss:.4f}-train_acc-{acc:.4f}" \
            "-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['loss', 'val_loss', 'acc', 'val_acc'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=2)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0)


initial_epoch = 0
ckpt_period = 10
n_epochs_to_train = 1000
ckpts = glob.glob("checkpoints/*.hdf5")

if len(ckpts) != 0:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint: ", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.find("-train")])
    model = load_model(latest_ckpt)
else:
    model = CnnModel().build()

optimizer = Adam(0.000006)
model = multi_gpu_model(model, gpus=8)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
model.fit_generator(
        train_generator,
        steps_per_epoch=315 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=90 // batch_size,
        initial_epoch=initial_epoch,
        workers=12,
        callbacks=[checkpoint, tensorboard])

