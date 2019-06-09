"""
This is for Siamese model to remove corrupted npy files
"""

import numpy as np
import os
from multiprocessing import Pool
thread_count = 4
data_directory = 'dataset/raw_data'


def delete(category_dir):
    all_files = [os.path.join(category_dir, x) for x in os.listdir(category_dir)]
    print(all_files)
    for file in all_files:
        try:
            a = np.load(file)
            if a.shape != (2,150,150,3):
                raise Exception
        except:
            print('Error reading the file {0}'.format(file))
            os.remove(file)


def delete_many(category_dir):
    all_files = [os.path.join(category_dir, x) for x in os.listdir(category_dir)]
    for file in all_files[10:]:
        os.remove(file)


if __name__ == '__main__':
    directories = ['dataset/train_npy/similar',
                   'dataset/train_npy/different',
                   'dataset/val_npy/similar',
                   'dataset/val_npy/different']
    pool = Pool(thread_count)
    pool.map(delete_many, directories)

# delete('dataset_old/train_npy/similar')