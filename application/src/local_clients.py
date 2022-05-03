import pickle
from logging import INFO

import flwr as fl
import os
import gc
import gridfs
import pandas as pd
import tensorflow as tf
from flwr.common.logger import log
from keras import Sequential
from keras.constraints import maxnorm
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    client = LOCifarClient()
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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 8333 : (idx + 1) * 8333],
        y_train[idx * 8333 : (idx + 1) * 8333],
    ), (
        x_test[idx * 1666 : (idx + 1) * 1666],
        y_test[idx * 1666 : (idx + 1) * 1666],
    )


class LOLeukemiaClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = int(os.getenv('USER_INDEX'))
        self.losses = []
        self.accuracies = []
        self.precisions = []
        all_training = os.walk(os.path.join(os.sep, 'data', 'new_split', f'fold_{self.index}'))
        img_paths = []
        labels = []
        self.new_labels = []
        for d in all_training:
            if "all" in d[0]:
                for img_name in d[2]:
                    img_paths.append(os.path.join(d[0], img_name))
                    labels.append("0")
            elif "hem" in d[0]:
                for img_name in d[2]:
                    img_paths.append(os.path.join(d[0], img_name))
                    labels.append("1")
        self.img_paths, self.labels = np.array(img_paths), np.array(labels)
        d = {'images': self.img_paths, 'labels': labels}
        self.data = pd.DataFrame(data=d)
        aug_model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1)
        ])
        self.model = get_cnn_model_1(IMAGE_SIZE + (3,))
        adam_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True)
        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy", "precision"]
        self.core_idg = ImageDataGenerator(horizontal_flip=True,
                                           vertical_flip=True,
                                           brightness_range=[0.9, 1.0],
                                           validation_split=0.2)
        self.model.compile(
            optimizer=adam_opt,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        self.train_gen = self.core_idg.flow_from_directory(os.path.join(os.sep, 'data', 'new_split', f'fold_{self.index}'),
                                                 class_mode='categorical',
                                                 target_size=IMAGE_SIZE,
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 subset='training'
                                                      )
        self.valid_gen = self.core_idg.flow_from_directory(os.path.join(os.sep, 'data', 'new_split', f'fold_{self.index}'),
                                                 class_mode='categorical',
                                                 target_size=IMAGE_SIZE,
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 subset='validation')
        self.steps, self.val_steps = self.train_gen.samples, self.valid_gen.samples

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.train_gen,
            steps_per_epoch=self.steps//BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=self.valid_gen,
            validation_steps=self.val_steps//BATCH_SIZE,
        )
        self.valid_gen.reset()
        self.train_gen.reset()
        return self.model.get_weights(), len(self.img_paths), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision = model.evaluate(self.valid_gen, steps=self.val_steps // BATCH_SIZE)
        losses, lengths, metrics = loss, len(self.val_steps), {'accuracy': accuracy, 'precision': precision}
        self.valid_gen.reset()
        self.losses.append(losses)
        self.precisions.append(metrics["precision"])
        self.accuracies.append(metrics["accuracy"])
        with open(os.path.join(os.sep, "code", "application", "results.pkl"), 'wb') as handle:
            results = {"losses": self.losses, "precision": self.precisions, "accuracy": self.accuracies}
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return losses, lengths, metrics


class LOCifarClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = os.getenv('USER_INDEX')
        log(INFO, f"Local index is {self.index}")
        self.losses = []
        self.accuracies = []
        self.precisions = []
        # Load a subset of CIFAR-10 to simulate the local data partition
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
        self.model.add(Dense(self.num_classes, activation='softmax'))
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
            validation_split=0.1,
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
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 16)
        num_examples_test = len(self.x_test)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        with open(os.path.join(os.sep, "code", "application", "results.pkl"), 'wb') as handle:
            results = {"loss": self.losses, "accuracy": self.accuracies}
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return loss, num_examples_test, {"accuracy": accuracy}
