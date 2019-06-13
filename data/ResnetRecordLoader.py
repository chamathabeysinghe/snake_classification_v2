import numpy as np
import keras
import os


class ResnetRecordLoader(keras.utils.Sequence):

    def __init__(self, path, batch_size, shuffle=True):
        self.batch_size = batch_size
        sub_dirs = [x for x in os.listdir(path) if not x.startswith('.')]
        sub_dirs = sorted(sub_dirs)
        file_names = []
        for index, sub_dir in enumerate(sub_dirs):
            category = np.zeros(len(sub_dirs))
            category[index] = 1
            files = [(os.path.join(path, sub_dir, x), category) for x in os.listdir(os.path.join(path, sub_dir))
                     if not x.startswith('.')]
            file_names += files
        self.file_names = file_names
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        selected_files = self.file_names[index * self.batch_size: (index + 1) * self.batch_size]
        y = np.asarray([label for _, label in selected_files])
        X = [file for file, _ in selected_files]
        X = self.__data_generation(X)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_names)

    def __data_generation(self, files):
        data = []
        for file in files:
            data.append(np.load(file))
        return np.asarray(data)
