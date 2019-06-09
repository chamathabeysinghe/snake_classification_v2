"""
Generates npys for Siamese network
"""
import os
import itertools
from PIL import Image
# from skimage import io
from skimage import transform
import numpy as np
import uuid
from multiprocessing import Pool


npy_dir = '/home/ubuntu/dataset/val_npy'
similar_dir = os.path.join(npy_dir, 'similar')
different_dir = os.path.join(npy_dir, 'different')
image_data_dir = '/home/ubuntu/dataset/val'
thread_count = 64


def save_pair(pair, save_dir):
    try:
        img_1 = Image.open(pair[0])
        if img_1.mode != 'RGB':
            img_1 = img_1.convert('RGB')
        img_1 = np.asarray(img_1)
        img_1 = transform.resize(img_1, (150, 150)) - 0.5

        img_2 = Image.open(pair[1])
        if img_2.mode != 'RGB':
            img_2 = img_2.convert('RGB')
        img_2 = np.asarray(img_2)
        img_2 = transform.resize(img_2, (150, 150)) - 0.5

        # img_1 = io.imread(pair[0])
        # img_2 = io.imread(pair[1])
        # img_1 = transform.resize(img_1, (150, 150), anti_aliasing=True) - 0.5
        # img_2 = transform.resize(img_2, (150, 150), anti_aliasing=True) - 0.5
        out_array = np.stack([img_1, img_2], axis=0)
        np.save(os.path.join(save_dir, uuid.uuid4().hex), out_array)
    except:
        print('Error occurred reading the pair {0}'.format(str(pair)))
        return


def generate_similar_pairs(category_dir):
    images = [os.path.join(category_dir, x) for x in os.listdir(category_dir) if not x.startswith('.')]
    images = images[0:min(60, len(images))]
    print(category_dir)
    for pair in itertools.combinations(images, 2):
        save_pair(pair, similar_dir)
    print('{0} category done'.format(category_dir))


def generate_different_pairs(category_dir_pair):
    images_1 = [os.path.join(category_dir_pair[0], file) for file in os.listdir(category_dir_pair[0])
                if not file.startswith('.')]
    images_1 = images_1[0: min(9, len(images_1))]
    images_2 = [os.path.join(category_dir_pair[1], file) for file in os.listdir(category_dir_pair[1])
                if not file.startswith('.')]
    images_2 = images_2[0: min(9, len(images_2))]
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
    directory_pairs = [l for l in itertools.combinations(directories, 2)]
    pool = Pool(thread_count)
    pool.map(generate_different_pairs, directory_pairs)
    print('All pairs were done')


if __name__ == '__main__':
    # make_similar_pairs(image_data_dir)
    make_different_pairs(image_data_dir)
