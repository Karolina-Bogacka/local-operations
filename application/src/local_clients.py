import pickle
from logging import INFO

import flwr as fl
import os
import gc
import gridfs
import pandas as pd
import tensorflow as tf
from PIL import Image
from flwr.common.logger import log
from keras import Sequential
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from starlette.concurrency import run_in_threadpool
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import VGG16

from application.config import DB_PORT, FEDERATED_PORT, DATABASE_NAME
from application.utils import formulate_id

current_jobs = {}
ERAS = 50
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




def get_ds(filenames, labels, batch_size, pref_buf_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    label_ds, image_pathes = tf.data.Dataset.from_tensor_slices(labels), tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = image_pathes.map(load_image, AUTOTUNE).map(preprocess, AUTOTUNE)
    ds = tf.data.Dataset.zip((images_ds, label_ds)).batch(batch_size).prefetch(pref_buf_size)
    return ds


def load_image(path):
    image = tf.io.decode_bmp(tf.io.read_file(path), channels=3)
    return image


def preprocess(image):
    result = tf.image.resize(image, (128, 128))
    result = tf.image.per_image_standardization(result)
    return result


def augment(image):
    max_gamma_delta = 0.1
    image = tf.image.random_brightness(image, max_delta=max_gamma_delta, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_flip_left_right(image, seed=SEED)
    return image


async def start_client(id, config):
    client = LOGermanClient()
    await run_in_threadpool(
        lambda: fl.client.start_numpy_client(server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client))
    current_id = formulate_id(config)
    if current_id in current_jobs and current_jobs[current_id] > 1:
        current_jobs[current_id] -= 1
    else:
        current_jobs.pop(current_id)


# Define local client
class LOKerasClient(fl.client.NumPyClient):

    def __init__(self, config):
        self.priv_config = config
        client = MongoClient(DATABASE_NAME, DB_PORT)
        db = client.local
        db_grid = client.repository_grid
        fs = gridfs.GridFS(db_grid)
        if db.models.find_one({"id": config.model_id, "version": config.model_version}):
            result = db.models.find_one({"id": config.model_id, "version": config.model_version})
            self.model = pickle.loads(fs.get(result['model_id']).read())
            self.model.__init__(config.shape, classes=config.num_classes, weights=None)
        else:
            self.model = tf.keras.applications.MobileNetV2(config.shape, classes=config.num_classes, weights=None)
        self.model.compile(config.optimizer, config.eval_func, metrics=config.eval_metrics)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = config["config"][0].epochs if config else self.priv_config.config[0].epochs
        batch_size = int(config["config"][0]["batch_size"]) if config else self.priv_config.config[0].batch_size
        steps_per_epoch = config["config"][0].steps_per_epoch if config else self.priv_config.config[0].steps_per_epoch
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, metrics = self.model.evaluate(self.x_test, self.y_test)
        if not isinstance(metrics, list):
            metrics = [metrics]
        evaluations = {m: metrics[i] for i, m in enumerate(self.priv_config.eval_metrics)}
        return loss, len(self.x_test), evaluations


def load_partition(idx: int):
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
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1,
                                                        random_state=42)

    # Displaying the shape after the split
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return (X_train, y_train), (X_test, y_test)





class LOGermanClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = os.getenv('USER_INDEX')
        log(INFO, f"Local index is {self.index}")
        self.losses = []
        self.accuracies = []
        self.precisions = []
        # Load a subset of CIFAR-10 to simulate the local data partition
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
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])

        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy", "precision"]


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            32,
            epochs=1,
            validation_data=(self.x_test, self.y_test)
        )
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        num_examples_test = len(self.x_test)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        with open(os.path.join(os.sep, "code", "application", "results.pkl"), 'wb') as handle:
            results = {"loss": self.losses, "accuracy": self.accuracies}
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return loss, num_examples_test, {"accuracy": accuracy}
