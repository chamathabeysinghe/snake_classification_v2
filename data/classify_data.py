from os import listdir
from os import makedirs
from os import path
import random
import shutil
import math
from multiprocessing import Pool

data_directory = 'dataset/raw_data'
train_directory = 'dataset/train'
val_directory = 'dataset/val'

thread_count = 8


def split_data(directory_name):
    base_path = path.join(data_directory, directory_name)
    train_path = path.join(train_directory, directory_name)
    val_path = path.join(val_directory, directory_name)
    makedirs(train_path, exist_ok=True)
    makedirs(val_path, exist_ok=True)

    all_images = listdir(base_path)
    random.shuffle(all_images)
    count = len(all_images)

    for i in range(math.ceil(count * 0.75)):
        file = all_images[i]
        shutil.copy(path.join(base_path, file), path.join(train_path, file))
    for j in range(math.ceil(count * 0.75), count):
        file = all_images[j]
        shutil.copy(path.join(base_path, file), path.join(val_path, file))


if __name__ == '__main__':
    makedirs(train_directory, exist_ok=True)
    makedirs(val_directory, exist_ok=True)
    directories = [x for x in listdir(data_directory) if not x.startswith('.')]
    pool = Pool(thread_count)
    pool.map(split_data, directories)

