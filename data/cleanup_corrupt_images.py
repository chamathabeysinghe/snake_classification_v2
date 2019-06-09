"""
Remove corrupted data from a directory
"""
from skimage import io
import os
from multiprocessing import Pool
thread_count = 8
data_directory = 'dataset/raw_data'


def split_data(category_dir):
    all_images = [os.path.join(category_dir, x) for x in os.listdir(category_dir)]
    for img in all_images:
        try:
            io.imread(img)
        except:
            print('Error reading the image {0}'.format(img))
            os.remove(img)


if __name__ == '__main__':
    directories = [os.path.join(data_directory, x) for x in os.listdir(data_directory) if not x.startswith('.')]
    pool = Pool(thread_count)
    pool.map(split_data, directories)
