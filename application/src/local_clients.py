import os
import pickle

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from logging import WARNING, INFO
import time

from PIL import Image
from flwr.common.logger import log
from flwr.common import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    Scalar,
    Status,
    parameters_to_weights,
    weights_to_parameters,
)
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export

from sklearn.model_selection import StratifiedKFold, train_test_split
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from application.config import DB_PORT, FEDERATED_PORT, DATABASE_NAME
from application.my_client import MyClient
from application.my_client_start import start_my_client
from application.utils import formulate_id


current_jobs = {}
ERAS = 1
EPOCHS = 1
SEED = 42
BATCH_SIZE = 16
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




def start_client(id, cluster_index, config, losses, priority, queue, queue_loss,
                 queue_cluster, accuracies, times):
    client = LOCifarClient(cluster_index, losses, priority, config, queue,
                              queue_loss, queue_cluster, accuracies, times)
    start_my_client(server_address=f"{config.server_address}"
                                                 f":{FEDERATED_PORT}", client=client)
    current_id = formulate_id(config)
    if current_id in current_jobs and current_jobs[current_id] > 1:
        current_jobs[current_id] -= 1
    else:
        current_jobs.pop(current_id)

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
    """Load 1/8th of the training and test data to simulate a partition."""
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
            path = os.path.join(os.sep, 'german-traffic', 'Train', str(i))
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

class LOCifarClient(MyClient):

    def __init__(self, cluster_index, losses, priority, config, queue, queue_loss,
                 queue_cluster, accuracies, times):
        self.queue = queue
        self.queue_loss = queue_loss
        self.queue_cluster = queue_cluster
        self.accuracies = accuracies
        self.times = times
        self.priv_config = config
        self.priority = priority
        self.cluster_index = cluster_index
        self.index = os.getenv('USER_INDEX')
        self.losses_visual = []
        self.precisions = []
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
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])

        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy"]
        self.datagen = ImageDataGenerator()
        self.local_gen = self.datagen.flow(self.x_train, self.y_train,
                                           batch_size=BATCH_SIZE)
        self.losses = losses
        self.assigned_cluster = -1
        self.lowest_epochs = 10
        self.model_weights = {}
        self.current_epoch = 0
        self.current_step = 0
        self.step_diff = 12
        self.possible_steps = len(self.x_train)//(BATCH_SIZE*self.step_diff)



    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Return the current client properties."""
        return PropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={"cluster_index": self.cluster_index},
        )

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        log(INFO, f"Parameter length is {len(parameters)}")
        self.model.set_weights(parameters)
        self.lowest_epochs = config["local_epochs"]
        log(INFO,
            f"At the beginning of fit losses of client{os.environ['USER_INDEX']} "
            f"look like this {np.array(self.losses)}")
        log(INFO, f'Received config of model:{config["model_index"]}')
        log(INFO, f'Current priority of :{self.priority.value}')

        if self.current_step >= self.possible_steps:
            self.current_step = 0
            self.local_gen.reset()

        if self.current_epoch <= config["local_epochs"]:
            log(INFO, f'This local client is of index:{self.cluster_index}')
            self.model_weights[config['model_index']] = parameters
            log(INFO, f'Iteration of index {self.current_epoch}')
            history = self.model.fit(
                self.local_gen,
                batch_size=BATCH_SIZE,
                epochs=5,
            )
            results = {
                "loss": history.history["loss"][0],
                "accuracy": history.history["accuracy"][0],
            }
            log(INFO, f"{results}")
            log(INFO, f"Loss of model {config['model_index']} equal to {history.history['loss']}")
            log(INFO,
                f"Client of index {config['model_index']} equal to"
                f" {history.history['loss']}")

            if self.priority.value == self.priv_config.num_clusters-1:
                log(INFO,
                    f"Client of index {config['model_index']} currently has all the "
                    f"current info")

                lowest_loss = history.history['loss'][0]
                lowest_cluster = config["model_index"]
                lowest_weights = self.model.get_weights()
                lowest_steps = self.step_diff*BATCH_SIZE
                while not self.queue.empty():
                    weights, loss, index, steps = self.queue.get()
                    self.priority.value -= 1
                    if loss < lowest_loss:
                        lowest_cluster = index
                        lowest_weights = weights
                        lowest_loss = loss
                        lowest_steps = steps
                #self.model.set_weights(self.model_weights[index_of_lowest])
                log(INFO,
                    f"At the end of fit losses of {os.environ['USER_INDEX']} "
                    f"look like this {np.array(self.losses)}")
                self.current_epoch += 1
                self.current_step += self.step_diff
                self.queue_cluster.put((lowest_weights, lowest_cluster))
                self.assigned_cluster = lowest_cluster
                return lowest_weights, lowest_steps, {
                    "loss": lowest_loss, "assigned_cluster": lowest_cluster,
                    "clustering_phase":
                        self.current_epoch < config["local_epochs"], "failure": False}
            else:
                self.current_epoch += 1
                self.current_step += self.step_diff
                self.priority.value += 1
                self.queue.put((self.model.get_weights(), history.history['loss'][0],
                                config["model_index"], self.step_diff*BATCH_SIZE))
                return self.model.get_weights(), -1, \
                       {"failure": True, "clustering_phase":
                        self.current_epoch < config["local_epochs"],}
        else:
            log(INFO, f"This client's iterative phase assigned it to "
                      f":{config['model_index']}")
            log(INFO, f'Iteration of index {self.current_epoch}')
            history = self.model.fit(
                self.local_gen,
                batch_size=BATCH_SIZE,
                steps_per_epoch= len(self.x_train)//BATCH_SIZE,
                epochs=1,
            )
            results = {
                "loss": history.history["loss"][0],
                "accuracy": history.history["accuracy"][0]
            }
            log(INFO, f"{results}")
            self.current_epoch += 1
            self.current_step += self.step_diff

            return self.model.get_weights(), len(self.x_train), {"loss": history.history[
                'loss'][0], "assigned_cluster": self.assigned_cluster, "clustering_phase":
                self.current_epoch < config["local_epochs"], "failure": False}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        losses, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        lengths = len(self.x_test)
        log(INFO, f"Metrics {accuracy}")
        self.losses[self.current_epoch - 1] = losses
        self.accuracies[self.current_epoch - 1] = accuracy
        self.times[self.current_epoch - 1] = time.time()
        with open(os.path.join(os.sep, "code", "application", "results.pkl"),
                  'wb') as handle:
            results = {"loss": np.array(self.losses), "accuracy":
                np.array(self.accuracies),
                       "times": np.array(self.times)}
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.model.save(os.path.join(os.sep, "code", "application", "model"))
        log(INFO, f"Client {self.index} is in cluster {self.assigned_cluster} with "
                  f"accuracy {accuracy}")
        metrics = {"accuracy": accuracy}
        return losses, lengths, metrics
