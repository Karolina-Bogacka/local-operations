import os
import pickle

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from logging import WARNING, INFO
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

from sklearn.model_selection import StratifiedKFold
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
IMAGE_SIZE = (128, 128)
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
        loss, accuracy, precision = model.evaluate(test_gen, steps=len(filenames))
    else:
        loss, accuracy, precision = model.evaluate(test_gen, steps=len(filenames), callbacks=callbacks)
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


def load_image(path):
    image = tf.io.decode_bmp(tf.io.read_file(path), channels=3)
    return image


def preprocess(image):
    result = tf.image.resize(image, (128, 128))
    result = tf.image.per_image_standardization(result)
    return result


def start_client(id, cluster_index, config, losses, priority, queue, queue_loss,
                 queue_cluster):
    client = LOLeukemiaClient(cluster_index, losses, priority, config, queue,
                              queue_loss, queue_cluster)
    start_my_client(server_address=f"{config.server_address}"
                                                 f":{FEDERATED_PORT}", client=client)
    current_id = formulate_id(config)
    if current_id in current_jobs and current_jobs[current_id] > 1:
        current_jobs[current_id] -= 1
    else:
        current_jobs.pop(current_id)


class LOLeukemiaClient(MyClient):

    def __init__(self, cluster_index, losses, priority, config, queue, queue_loss,
                 queue_cluster):
        self.queue = queue
        self.queue_loss = queue_loss
        self.queue_cluster = queue_cluster
        self.priv_config = config
        self.priority = priority
        self.cluster_index = cluster_index
        self.index = os.getenv('USER_INDEX')
        self.losses_visual = []
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
        adam_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True)
        metrics = ["accuracy", tf.keras.metrics.Precision(name="precision")]
        self.metric_names = ["accuracy", "precision"]
        self.core_idg = ImageDataGenerator(horizontal_flip=True,
                                           vertical_flip=True,
                                           brightness_range=[0.9, 1.0])
        self.model.compile(
            optimizer=adam_opt,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        self.local_gen = self.core_idg.flow_from_dataframe(dataframe=self.data,
                                                           directory=os.path.join(os.sep,
                                                                                  'data',
                                                                                  'new_split',
                                                                                  f'fold_{self.index}'),
                                                           x_col='images',
                                                           y_col='labels',
                                                           class_mode='binary',
                                                           target_size=IMAGE_SIZE,
                                                           color_mode='rgb',
                                                           batch_size=BATCH_SIZE)
        self.losses = losses
        self.assigned_cluster = -1
        self.model_weights = {}
        self.current_epoch = 0
        self.current_step = 0
        self.step_diff = 12
        self.possible_steps = len(self.data)//(BATCH_SIZE*self.step_diff)

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
                steps_per_epoch=self.step_diff,
                epochs=1
            )
            log(INFO, f"Loss of model {config['model_index']} equal to {history.history['loss']}")
            self.losses[config["model_index"]] = history.history['loss'][0]
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
                       {"failure": True}
        else:
            log(INFO, f"This client's iterative phase assigned it to "
                      f":{config['model_index']}")
            log(INFO, f'Iteration of index {self.current_epoch}')
            history = self.model.fit(
                self.local_gen,
                batch_size=BATCH_SIZE,
                steps_per_epoch=self.step_diff,
                epochs=1
            )
            self.current_epoch += 1
            self.current_step += self.step_diff

            return self.model.get_weights(), self.step_diff*BATCH_SIZE, {"loss": history.history[
                'loss'][0], "assigned_cluster": self.assigned_cluster, "clustering_phase":
                self.current_epoch < config["local_epochs"], "failure": False}

    def evaluate(self, parameters, config):
        losses, lengths, metrics = 0.0, 10, {m: 0.0 for m in self.metric_names}
        log(INFO, f"Metrics {metrics}")
        if not self.queue_cluster.empty():
            results = self.queue_cluster.get()
            weights, cluster_index = results
            self.model.set_weights(weights)
            losses, lengths, metrics = test_model(self.model)
            self.losses_visual.append(losses)
            self.precisions.append(metrics["precision"])
            self.accuracies.append(metrics["accuracy"])
            log(INFO, f"Client {self.index} is in cluster {cluster_index} with "
                         f"accuracy {metrics['accuracy']}")
        return losses, lengths, metrics
