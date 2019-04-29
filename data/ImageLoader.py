import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from PIL import Image
from skimage import transform

h = 150
w = 150
d = 3


class ImageLoader:

    def __init__(self, path):
        self.data = {}
        self.categories = {}
        data_subsets = ['train', 'val']
        for name in data_subsets:
            X, y = self.load_images(os.path.join(path, name))
            self.data[name] = X
            self.categories[name] = y
            print('Loading {}'.format(name))
            print('X shape is {}'.format(X.shape))
            print('y shape is {}'.format(y.shape))

    @staticmethod
    def load_images(path):
        X = []
        y = []
        # we load every alphabet seperately so we can isolate them later
        for breed_id in [x for x in os.listdir(path) if not x.startswith('.')]:
            print("loading snake breed: " + breed_id)
            # lang_dict[alphabet] = [curr_y,None]
            # alphabet_path = os.path.join(path,alphabet)
            # every letter/category has it's own column in the array, so  load seperately
            breed_path = os.path.join(path, breed_id)
            breed_images = []
            for filename in [x for x in os.listdir(breed_path) if not x.startswith(".")]:
                image_path = os.path.join(breed_path, filename)
                # image = imread(image_path)
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.asarray(image)
                image = transform.resize(image, (w, h))
                breed_images.append(image)
            breed_images = np.asarray(breed_images)
            X.append(breed_images)
            y.append(breed_id)

            # if int(breed_id) == 46:
            #     break

        X = [x for _, x in sorted(zip(y, X))]
        y = [y for y, _ in sorted(zip(y, X))]
        y = np.vstack(y)
        X = np.asarray(X)
        return X, y

    def get_batch(self, batch_size, s='train'):
        X = self.data[s]
        n_classes = X.shape[0]

        categories = rnd.choice(n_classes, size=(batch_size,), replace=True)

        pairs = [np.zeros((batch_size, h, w, d)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1

        for i in range(batch_size):
            category_1 = categories[i]
            idx_1 = rnd.randint(0, X[category_1].shape[0])
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category_1
                idx_2 = (idx_1 + rnd.randint(1, X[category_2].shape[0])) % X[category_2].shape[0]
            else:
                category_2 = (category_1 + rnd.randint(1, n_classes)) % n_classes
                idx_2 = rnd.randint(0, X[category_2].shape[0])

            pairs[0][i, :, :, :] = X[category_1][idx_1].reshape((w, h, d))
            pairs[1][i, :, :, :] = X[category_2][idx_2].reshape((w, h, d))

        return pairs, targets
        # return np.asarray(pairs), np.asarray(targets)

    def get_validation_batch(self, batch_size, s='val'):
        x_val = self.data['val']
        x = self.data['train']
        n_classes = x_val.shape[0]

        categories = rnd.choice(n_classes, size=(batch_size,), replace=True)

        pairs = [np.zeros((batch_size, h, w, d)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1

        for i in range(batch_size):
            category_1 = categories[i]
            idx_1 = rnd.randint(0, x_val[category_1].shape[0])
            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category_1
                idx_2 = rnd.randint(0, x[category_2].shape[0])
            else:
                category_2 = (category_1 + rnd.randint(1, n_classes)) % n_classes
                idx_2 = rnd.randint(0, x[category_2].shape[0])

            pairs[0][i, :, :, :] = x_val[category_1][idx_1].reshape((w, h, d))
            pairs[1][i, :, :, :] = x[category_2][idx_2].reshape((w, h, d))

        return pairs, targets

    def make_oneshot_task(self, test_case_id=None):
        """Create a one-shot classification task
        pairs = <one image from validation set, images in training for all the classes>
        targets = <single '1' and all others are '0'>
        """
        x_val = self.data['val']
        y_val = self.categories['val']
        n_classes = x_val.shape[0]
        test_index =  test_case_id if test_case_id else rnd.randint(0, n_classes)
        test_category = y_val[test_index]

        test_image = x_val[test_index][0]
        test_image_set = [test_image for _ in range(n_classes)]

        support_set = []
        x_train = self.data['train']
        for i in range(len(x_train)):
            support_set.append(x_train[i][0])

        targets = np.zeros((n_classes,))
        targets[test_index] = 1

        pairs = [np.asarray(test_image_set), np.asarray(support_set)]
        return pairs, targets

    def make_whole_validation_set(self):
        test_set = []
        sample_set = []
        targets = []
        for k in range(0,84):
            print(k)
            pairs, targetforpairs = self.make_oneshot_task(k)
            test_set.append(pairs[0])
            sample_set.append(pairs[1])
            targets.append(targetforpairs)
        test_set = np.concatenate(test_set)
        sample_set = np.concatenate(sample_set)
        targets = np.concatenate(targets)
        return [test_set, sample_set], targets

    def make_human_test_couple(self, _id, expected_target):
        x_val = self.data['val']
        x_train = self.data['train']


    def generate(self, batch_size, s="train"):
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            # pairs = [(12,21) for _ in range(batch_size)]
            # targets = [0 for _ in range(batch_size)]
            yield pairs, targets

    def generate_val(self, batch_size):
        while True:
            pairs, targets = self.get_validation_batch(batch_size)
            yield pairs, targets

    def test_oneshot(self, model,k):
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task()
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        print("Got an average of {}% one-shot learning accuracy".format(percent_correct))
        return percent_correct


# def illustrate_dataset(pairs, batch_size, targets):
#     fig, axarr = plt.subplots(1, 2)
#
#     for i in range(batch_size):
#         title = ''
#         if (targets[i]==1):
#             title = 'Matching'
#         else:
#             title = 'Not matching'
#
#         fig.suptitle(title, fontsize=16)
#
#         img1 = pairs[0][i]
#         img2 = pairs[1][i]
#         axarr[0].imshow(img1)
#         axarr[1].imshow(img2)
#         # plt.show()
#         plt.waitforbuttonpress()
#
#
# loader = SiameseLoader('./data')
# pairs, targets = loader.get_validation_batch(32)
#
# illustrate_dataset(pairs, 32, targets)
