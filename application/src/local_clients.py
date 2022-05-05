import pickle
import time
from logging import INFO

import flwr as fl
import os
import gc
import pandas as pd
import tensorflow as tf
import numpy as np
from flwr.client import start_numpy_client
from flwr.common.logger import log
from keras import backend, Sequential
from keras.constraints import maxnorm
from keras.datasets.cifar import load_batch
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
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



def start_client(id, config):
    log(INFO, f"{config.server_address}:8081")
    client = LOCifarClient()
    start_numpy_client(server_address=f"{config.server_address}:8081",
                                 client=client)

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
    (x_train, y_train), (x_test, y_test) = load_data()
    return (
        x_train[idx * 8333 : (idx + 1) * 8333],
        y_train[idx * 8333 : (idx + 1) * 8333],
    ), (
        x_test[idx * 1666 : (idx + 1) * 1666],
        y_test[idx * 1666 : (idx + 1) * 1666],
    )

class LOCifarClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = os.getenv('USER_INDEX')
        log(INFO,
            f"The loweliest client of {self.index} is in init")
        self.losses = []
        self.accuracies = []
        self.times = []
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_partition(
            int(self.index))
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu',
                              padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])

        self.metric_names = ["accuracy"]
        self.datagen = ImageDataGenerator()
        self.local_gen = self.datagen.flow(self.x_train, self.y_train,
                                           batch_size=BATCH_SIZE)
        self.current_epoch = 0
        self.current_step = 0
        self.step_diff = 1
        self.possible_steps = len(self.x_train)//(BATCH_SIZE*self.step_diff)
        log(INFO,
            f"The loweliest client of {self.index} has finished init with data of "
            f"length {len(self.x_train)}")


    def get_parameters(self):
        log(INFO, "Returned parameters")
        return self.model.get_weights()

    def get_properties(self, config):
        properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        return properties

    def fit(self, parameters, config):
        log(INFO, f"Parameter length is {len(parameters)}")
        self.model.set_weights(parameters)
        log(INFO,
            f"The loweliest client of {self.index} has began the fit")

        if self.current_step >= self.possible_steps:
            self.current_step = 0
            self.local_gen.reset()

        log(INFO, f'Iteration of index {self.current_epoch}')
        history = self.model.fit(
            self.local_gen,
            batch_size=BATCH_SIZE,
            steps_per_epoch=self.step_diff,
            epochs=1,
        )
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }
        log(INFO, f"{results}")
        log(INFO, f"Loss of model {self.index} equal to {history.history['loss']}")
        return self.model.get_weights(), BATCH_SIZE, {}


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
