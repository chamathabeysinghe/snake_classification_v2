from data.ResnetRecordLoader import ResnetRecordLoader
from matplotlib import pyplot as plt
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from model.ResnetModel import ResnetModel
from keras.optimizers import Adam
from time import time
import glob
from keras.models import load_model
from keras.utils import multi_gpu_model
from data.shared import base_directory

batch_size = 32
training_path = os.path.join(base_directory, 'train_resnet')
validation_path = os.path.join(base_directory, 'val_resnet')
train_generator = ResnetRecordLoader(training_path, batch_size)

validation_generator = ResnetRecordLoader(validation_path, batch_size)

os.makedirs("checkpoints_resnet", exist_ok=True)
file_path = "checkpoints_resnet/cnn-model-epoch-{epoch:05d}-train_loss-{loss:.4f}-train_acc-{acc:.4f}" \
            "-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['loss', 'val_loss', 'acc', 'val_acc'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=2)
tensorboard = TensorBoard(log_dir="logs_resnet/{}".format(time()), histogram_freq=0)


initial_epoch = 0
ckpt_period = 10
n_epochs_to_train = 1000
ckpts = glob.glob("checkpoints_resnet/*.hdf5")

if len(ckpts) != 0:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint: ", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.find("-train")])
    model = load_model(latest_ckpt)
else:
    model = ResnetModel().build()

optimizer = Adam(0.000006)
# model = multi_gpu_model(model, gpus=8)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        initial_epoch=initial_epoch,
        workers=12,
        callbacks=[checkpoint, tensorboard])

