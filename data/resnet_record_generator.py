from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
import numpy as np
import os
from data.shared import base_directory
from PIL import Image

input_dir = os.path.join(base_directory, 'val')
out_dir = os.path.join(base_directory, 'val_resnet')


model = InceptionResNetV2(weights='imagenet', include_top=True, input_shape=(299, 299, 3))
model = Model(inputs=model.input, outputs=model.layers[-2].output)


def write_files(predictions, file_names, sub_dir):
    os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)
    for i in range(len(file_names)):
        np.save(os.path.join(out_dir, sub_dir, file_names[i].split('.')[0]), predictions[i])


def get_predictions(images):
    images = preprocess_input(images)
    predictions = model.predict(images)
    return predictions


def get_files(cat_dir):
    print('Reading directory {0}'.format(cat_dir))
    file_names = [x for x in os.listdir(cat_dir) if not x.startswith('.')]
    files = [os.path.join(cat_dir, x) for x in file_names]
    images = []
    for file in files:
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((299, 299))
        img = np.asarray(img)
        images.append(img)
    images = np.asarray(images)
    return images, file_names


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    sub_dirs = [x for x in os.listdir(input_dir) if not x.startswith('.')]
    for sub_dir in sub_dirs:
        sub_dir_full = os.path.join(input_dir, sub_dir)
        images, file_names = get_files(sub_dir_full)
        predictions = get_predictions(images)
        write_files(predictions, file_names, sub_dir)
        print(predictions.shape)

