import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):

    def __init__(self, similar_path, different_path, similar_file_count, different_file_count, batch_size=32, h=150, w=150, channels=3, shuffle=True):
        self.similar_path = similar_path
        self.different_path = different_path
        self.similar_file_count = similar_file_count
        self.different_file_count = different_file_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h = h
        self.w = w
        self.channels = channels
        self.similar_files = np.arange(self.similar_file_count)
        self.different_files = np.arange(self.different_file_count)

    def __len__(self):
        """
        Number of batches per epoch
        """
        larger_count = max(self.similar_file_count, self.different_file_count)
        return int(np.floor(larger_count / self.batch_size))

    def __getitem__(self, index):
        half_batch_size = self.batch_size // 2
        similar_indexes = [x % self.similar_file_count for x in range(half_batch_size*index, half_batch_size*(index+1))]
        different_indexes = [x % self.different_file_count for x in range(half_batch_size*index, half_batch_size*(index+1))]

        similar_files = [self.similar_files[k] for k in similar_indexes]
        different_files = [self.different_files[k] for k in different_indexes]

        X, y = self.__data_generation(similar_files, different_files)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.similar_files)
            np.random.shuffle(self.different_files)

    def __data_generation(self, similar_ids, different_ids):
        X = [np.empty((self.batch_size, self.h, self.w, self.channels)) for _ in range(2)]
        y = np.empty(self.batch_size, dtype=int)

        for i, file_id in enumerate(similar_ids):
            full_path = np.load('{0}/{1}.npy'.format(self.similar_path, file_id))
            npy_record = np.load(full_path)
            X[0][i, ] = npy_record[0]
            X[1][i, ] = npy_record[1]
            y[i] = 1

        for j, file_id in enumerate(different_ids):
            full_path = np.load('{0}/{1}.npy'.format(self.similar_path, file_id))
            npy_record = np.load(full_path)
            X[0][j+self.batch_size//2, ] = npy_record[0]
            X[1][j+self.batch_size//2, ] = npy_record[1]
            y[i] = 1

        return X, y
