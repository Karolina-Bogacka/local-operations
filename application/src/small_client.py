import pickle
import time
from logging import INFO

import flwr as fl
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from flwr.client import start_numpy_client
from flwr.common.logger import log
from keras import backend, Sequential
from keras.constraints import maxnorm
from keras.datasets.cifar import load_batch
from keras.layers import MaxPooling2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

current_jobs = {}
ERAS = 1
EPOCHS = 60
SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = (32, 32)
PREFETCH_BUFFER_SIZE = 400
SHUFFLE_BUFFER_SIZE = 1000
CACHE_DIR = "caches/ds_cache"
ds_params = dict(
    labels="inferred",
    label_mode="categorical",
    class_names=["all", "hem"],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=SEED
)


def load_data():
    path = os.path.join(os.sep, "cifar-10-batches-py")
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    return (x_train, y_train), (x_test, y_test)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    """Load 1/10th of the training and test data to simulate a partition."""
    data = []
    labels = []
    classes = 43
    total_length = 0
    cur_path = os.getcwd()
    division = {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8, 9], 3: [10, 11],
                4: [12, 13, 14, 15, 16], 5: [17, 18, 19, 20, 21, 22, 23, 24],
                6: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                7: [35, 36, 37, 38, 39, 40, 41, 42]}
    # Retrieving the images and their labels
    for i in range(classes):
        if i in division[idx]:
            path = os.path.join('german-traffic', 'Train', str(i))
            images = os.listdir(path)
            for a in images:
                try:
                    image = Image.open(os.path.join(path, a))
                    image = image.resize((32, 32))
                    image = np.array(image)
                    data.append(image)
                    labels.append(i)
                except:
                    print("Error loading image")

    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)

    # Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=42)

    # Displaying the shape after the split
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return (X_train, y_train), (X_test, y_test)


class SmallCifarClient(fl.client.NumPyClient):

    def __init__(self, events):
        self.index = os.getenv('USER_INDEX')
        log(INFO,
            f"The loweliest client of {self.index} is in init")
        self.losses = []
        self.accuracies = []
        self.times = []
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_partition(
            int(self.index))
        self.num_classes = self.y_test.shape[1]
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                              input_shape=self.x_train.shape[1:]))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(43, activation='softmax'))

        # Compilation of the model
        #self.model.compile(loss='categorical_crossentropy', optimizer='adam',
        #                   metrics=['accuracy'])
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])

        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy", "precision"]
        self.datagen = ImageDataGenerator()
        self.local_gen = self.datagen.flow(self.x_train, self.y_train,
                                           batch_size=BATCH_SIZE)
        self.current_epoch = 0
        self.current_step = 0
        self.step_diff = 1
        self.possible_steps = len(self.x_train) // (BATCH_SIZE * self.step_diff)
        self.events = events
        self.current_round = 0
        self.lengths = 0

    def get_parameters(self):
        log(INFO, "Returned parameters")
        return self.model.get_weights()

    def get_properties(self, config):
        properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        return properties

    def fit(self, parameters, config):
        log(INFO, f"Parameter length is {len(parameters)}")
        self.model.set_weights(parameters)
        log(INFO, f'Iteration of index {self.current_round}')
        history = self.model.fit(
            self.local_gen,
            batch_size=BATCH_SIZE,
            steps_per_epoch=self.possible_steps,
            epochs=1,
        )
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }
        self.lengths = int(config["lengths"]) + len(self.x_train)
        self.events[self.current_round].set()
        log(INFO, f"Flag here set to {self.events[self.current_round].is_set()} with {results}")
        self.current_round += 1
        return self.model.get_weights(), self.lengths, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        losses, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        lengths = len(self.x_test)
        log(INFO, f"Metrics {accuracy}")
        self.losses.append(losses)
        self.accuracies.append(accuracy)
        self.times.append(time.time())
        with open(os.path.join(os.sep, "code", "application", "results.pkl"),
                  'wb') as handle:
            results = {"loss": self.losses, "accuracy":
                self.accuracies,
                       "times": self.times}
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metrics = {"accuracy": accuracy}
        return losses, lengths, metrics
