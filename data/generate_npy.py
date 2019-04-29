import os
import itertools
from skimage import io
from skimage import transform
import numpy as np
import uuid
from multiprocessing import Pool


npy_dir = 'dataset/val_npy'
similar_dir = os.path.join(npy_dir, 'similar')
different_dir = os.path.join(npy_dir, 'different')
image_data_dir = 'dataset/val'
thread_count = 8


def save_pair(pair, save_dir):
    try:
        img_1 = io.imread(pair[0])
        img_2 = io.imread(pair[1])
        img_1 = transform.resize(img_1, (150, 150), anti_aliasing=True) - 0.5
        img_2 = transform.resize(img_2, (150, 150), anti_aliasing=True) - 0.5
        out_array = np.stack([img_1, img_2], axis=0)
        np.save(os.path.join(save_dir, uuid.uuid4().hex), out_array)
    except:
        print('Error occurred reading the pair {0}'.format(str(pair)))
        return


def generate_similar_pairs(category_dir):
    images = [os.path.join(category_dir, x) for x in os.listdir(category_dir) if not x.startswith('.')]
    print(category_dir)
    for pair in itertools.product(images, images):
        save_pair(pair, similar_dir)
    print('{0} category done'.format(category_dir))


def generate_different_pairs(category_dir_pair):
    images_1 = [os.path.join(category_dir_pair[0], file) for file in os.listdir(category_dir_pair[0])
                if not file.startswith('.')]
    images_2 = [os.path.join(category_dir_pair[1], file) for file in os.listdir(category_dir_pair[1])
                if not file.startswith('.')]
    print(category_dir_pair)
    for pair in itertools.product(images_1, images_2):
        save_pair(pair, different_dir)
    print('{0} done'.format(str(category_dir_pair)))


def make_similar_pairs(data_dir):
    os.makedirs(similar_dir, exist_ok=True)
    directories = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if not x.startswith('.')]
    pool = Pool(thread_count)
    pool.map(generate_similar_pairs, directories)
    print('All categories were done')


def make_different_pairs(data_dir):
    os.makedirs(different_dir, exist_ok=True)
    directories = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if not x.startswith('.')]
    directory_pairs = [l for l in itertools.product(directories, directories) if l[0] != l[1]]
    pool = Pool(thread_count)
    pool.map(generate_different_pairs, directory_pairs)
    print('All pairs were done')


if __name__ == '__main__':
    # make_similar_pairs(image_data_dir)
    make_different_pairs(image_data_dir)
