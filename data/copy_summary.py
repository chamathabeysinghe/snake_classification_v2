"""
Create a small dataset from large dataset for testing
"""
import os
import shutil
from multiprocessing import Pool

source_dir = '/home/ubuntu/dataset/raw_data'
destination_dir = '/home/ubuntu/dataset/raw_data_small_dataset'


def extract_files(dir):
    files = os.listdir(os.path.join(source_dir, dir))
    category_dir_out = os.path.join(destination_dir, dir)
    os.makedirs(category_dir_out, exist_ok=True)
    for index in range(len(files)):
        if index > 8:
            break
        file = files[index]
        shutil.copy(os.path.join(source_dir, dir, file), os.path.join(category_dir_out, file))


os.makedirs(destination_dir, exist_ok=True)
directories = [x for x in os.listdir(source_dir) if not x.startswith('.')]
pool = Pool(32)
pool.map(extract_files, directories)
c