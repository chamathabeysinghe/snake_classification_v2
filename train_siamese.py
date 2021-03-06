import glob

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

from data.DataGenerator import DataGenerator
from data.ImageLoaderNPY import ImageLoader
from model.SiameseModel import SiameseModel
import os
from time import time
from keras.utils import multi_gpu_model

initial_epoch = 0
ckpt_period = 10
n_epochs_to_train = 1000
ckpts = glob.glob("checkpoints/*.hdf5")

if len(ckpts) != 0:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint: ", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.rfind("-lr-")])
    model = load_model(latest_ckpt)
else:
    model = SiameseModel().build()

optimizer = Adam(0.000000006)
model = multi_gpu_model(model, gpus=8)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

os.makedirs("checkpoints", exist_ok=True)
file_path = "checkpoints/siamese-epoch-{epoch:05d}-lr-" + "-train_loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['loss', 'val_loss'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='min',
                             period=ckpt_period)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0)

if n_epochs_to_train <= initial_epoch:
    n_epochs_to_train += initial_epoch

training_similar_path = '/home/ubuntu/dataset/train_npy/similar'
training_different_path = '/home/ubuntu/dataset/train_npy/different'
training_similar_file_count = len(os.listdir(training_similar_path))
training_different_file_count = len(os.listdir(training_different_path))

val_similar_path = '/home/ubuntu/dataset/val_npy/similar'
val_different_path = '/home/ubuntu/dataset/val_npy/different'
val_similar_file_count = len(os.listdir(val_similar_path))
val_different_file_count = len(os.listdir(val_different_path))

training_generator = DataGenerator(training_similar_path,
                                   training_different_path,
                                   training_similar_file_count,
                                   training_different_file_count,
                                   batch_size=500)

validation_generator = DataGenerator(val_similar_path,
                                     val_different_path,
                                     val_similar_file_count,
                                     val_different_file_count,
                                     batch_size=500)


model.fit_generator(generator=training_generator,
                    epochs=n_epochs_to_train,
                    validation_data=validation_generator,
                    callbacks=[checkpoint, tensorboard],
                    use_multiprocessing=True,
                    initial_epoch=initial_epoch,
                    workers=6)

# image_loader = ImageLoader('./data/dataset')
# model.fit_generator(generator=image_loader.generate(4),
#                     steps_per_epoch=1,
#                     epochs=n_epochs_to_train,
#                     validation_data=image_loader.generate_val(4),
#                     validation_steps=1,
#                     callbacks=[tensorboard],
#                     use_multiprocessing=True,
#                     initial_epoch=initial_epoch,
#                     workers=6)
