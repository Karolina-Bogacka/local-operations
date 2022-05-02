import pickle
from logging import INFO

import flwr as fl
import os
import gc
import pandas as pd
import tensorflow as tf
import numpy as np
from flwr.client import start_numpy_client
from flwr.common.logger import log
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import VGG16

from application.config import DB_PORT, FEDERATED_PORT, DATABASE_NAME
from application.src.server_client import NumPyClient
from application.utils import formulate_id

current_jobs = {}
ERAS = 1
EPOCHS = 1
SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
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


def get_cnn_model_1(input_shape):
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    # Adding flatten
    model.add(Flatten())

    # Adding full connected layer (dense)
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Adding output layer
    model.add(Dense(units=1, activation='sigmoid'))
    '''
    base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)
    '''

    return model


def test_model(model, callbacks=None):
    # ../input/leukemia-classification/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv
    test_dir = os.path.join(os.sep, 'data', 'C-NMC_Leukemia', "validation_data")
    test_data_csv = pd.read_csv(
        test_dir + "/C-NMC_test_prelim_phase_data_labels.csv"
    )
    # print(test_data_csv.head())
    # labels = np.array(test_data_csv["labels"].to_list())
    # inverted_labels = test_data_csv[["new_names", "labels"]].sort_values("new_names")["labels"].to_list()
    # labels = np.array([1 - label for label in inverted_labels])
    test_data_dir = test_dir + "/C-NMC_test_prelim_phase_data"
    dir_list = list(os.walk(test_data_dir))[0]
    filenames = sorted([test_data_dir + "/" + name for name in dir_list[2]])
    get_label_by_name = lambda x: test_data_csv.loc[test_data_csv['new_names'] == x]["labels"].to_list()[0]
    labels = [str(1 - get_label_by_name(name)) for name in dir_list[2]]
    idg = ImageDataGenerator()
    df = pd.DataFrame({'images': filenames, 'labels': labels})
    test_gen = idg.flow_from_dataframe(dataframe=df.head(200),
                                        directory=test_dir,
                                        x_col='images',
                                        y_col='labels',
                                        class_mode='binary',
                                        target_size=IMAGE_SIZE,
                                        color_mode='rgb',
                                        batch_size=1)
    #test_ds = get_ds(filenames, labels, BATCH_SIZE, PREFETCH_BUFFER_SIZE)
    if callbacks == None:
        loss, accuracy, precision = model.evaluate(test_gen, steps=200)
    else:
        loss, accuracy, precision = model.evaluate(test_gen, steps=200)
    return loss, len(filenames), {'accuracy': accuracy, 'precision': precision}


def get_stratified_data_gen(img_paths, labels, data, core_idg, index, num_splits=4):
    skf = StratifiedKFold(n_splits=4, random_state=SEED, shuffle=True)
    for train_index, test_index in skf.split(img_paths, labels):
        trainData = img_paths[train_index]
        testData = img_paths[test_index]
        ## create train, valid dataframe and thus train_gen , valid_gen for each fold-loop
        train_df = data.loc[data["images"].isin(list(trainData))]
        valid_df = data.loc[data["images"].isin(list(testData))]
        # create model object
        train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                                 directory=os.path.join(os.sep, 'data', 'new_split', f'fold_{index}'),
                                                 x_col='images',
                                                 y_col='labels',
                                                 class_mode='binary',
                                                 target_size=IMAGE_SIZE,
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE)
        valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                                 directory=os.path.join(os.sep, 'data', 'new_split', f'fold_{index}'),
                                                 x_col='images',
                                                 y_col='labels',
                                                 class_mode='binary',
                                                 target_size=IMAGE_SIZE,
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE)
        print("in gen")
        yield train_gen, valid_gen, len(trainData), len(testData)


def get_stratified_datasets(X, Y, n_splits=4):
    # Create Stratified object
    skf = StratifiedKFold(n_splits=4, random_state=SEED, shuffle=True)
    skf.get_n_splits(X, Y)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        p = np.random.permutation(len(X_train))
        X_train, y_train = X_train[p], y_train[p]
        yield [[X_train, y_train], [X_test, y_test]]


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


def start_client(id, config):
    log(INFO, f"{config.server_address}:8081")
    client = LOLeukemiaClient()
    start_numpy_client(server_address=f"{config.server_address}:8081",
                                 client=client)


class LOLeukemiaClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = os.getenv('USER_INDEX')
        log(INFO,
            f"The loweliest client of {self.index} is in init")
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
        self.model = get_cnn_model_1(IMAGE_SIZE + (3,))
        sgd_opt = tf.keras.optimizers.SGD(learning_rate=0.001)
        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy", "precision"]
        self.core_idg = ImageDataGenerator(horizontal_flip=True,
                                           vertical_flip=True,
                                           brightness_range=[0.9, 1.0])
        self.model.compile(
            optimizer=sgd_opt,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        self.local_gen = self.core_idg.flow_from_dataframe(dataframe=self.data,
                                                           x_col='images',
                                                           y_col='labels',
                                                           class_mode='binary',
                                                           target_size=IMAGE_SIZE,
                                                           color_mode='rgb',
                                                           batch_size=BATCH_SIZE)
        self.current_epoch = 0
        self.current_step = 0
        self.step_diff = 1
        self.possible_steps = len(self.data)//(BATCH_SIZE*self.step_diff)
        log(INFO,
            f"The loweliest client of {self.index} has finished init with data of "
            f"length {len(self.img_paths)}")


    def get_parameters(self):
        log(INFO, "Returned parameters")
        return self.model.get_weights()

    def get_properties(self, config):
        properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        return properties

    def fit(self, parameters, config):
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
            epochs=1
        )
        log(INFO, f"Loss of model equal to {history.history['loss']}")
        self.current_epoch += 1
        self.current_step += self.step_diff
        return self.model.get_weights(), self.step_diff*BATCH_SIZE, {"loss": history.history[
                'loss'][0]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        losses, lengths, metrics = 0.0, 10, {m: 0.0 for m in self.metric_names}
        log(INFO, f"Metrics {metrics}")
        if self.current_epoch % 20 == 0:
            losses, lengths, metrics = test_model(self.model)
            self.losses.append(losses)
            self.precisions.append(metrics["precision"])
            self.accuracies.append(metrics["accuracy"])
        return losses, lengths, metrics
