import pickle

import flwr as fl
import os
import gc
import gridfs
import pandas as pd
import tensorflow as tf
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


'''def test_model(model, callbacks=None):
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
    test_gen = idg.flow_from_dataframe(dataframe=df,
                                        directory=test_dir,
                                        x_col='images',
                                        y_col='labels',
                                        class_mode='binary',
                                        target_size=IMAGE_SIZE,
                                        color_mode='rgb',
                                        batch_size=BATCH_SIZE)
    # print(filenames)
    # print(test_data_csv[["new_names", "labels"]])
    #test_ds = get_ds(filenames, labels, BATCH_SIZE, PREFETCH_BUFFER_SIZE)
    if callbacks == None:
        loss, accuracy, precision = model.evaluate(test_gen, steps=len(filenames)//BATCH_SIZE)
    else:
        loss, accuracy, precision = model.evaluate(test_gen, steps=len(filenames)//BATCH_SIZE, callbacks=callbacks)
    return loss, len(filenames), {'accuracy': accuracy, 'precision': precision}'''


def get_stratified_data_gen(img_paths, labels, data, core_idg, index):
    print("in data gen")
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
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
    return


def get_stratified_datasets(X, Y):
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


async def start_client(id, config):
    client = LOLeukemiaClient()
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


class LOLeukemiaClient(fl.client.NumPyClient):

    def __init__(self):
        self.index = os.getenv('USER_INDEX')
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
                                                 class_mode='binary',
                                                 target_size=IMAGE_SIZE,
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 subset='training'
                                                      )
        self.valid_gen = self.core_idg.flow_from_directory(os.path.join(os.sep, 'data', 'new_split', f'fold_{self.index}'),
                                                 class_mode='binary',
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
        '''
        data_gen = get_stratified_datasets(self.img_paths, self.labels)
        while True:
            try:
                train_data, valid_data = next(data_gen)
                train_ds = get_ds(*train_data, BATCH_SIZE, PREFETCH_BUFFER_SIZE)
                train_ds = train_ds.map(lambda x, y: [augment(x), y], tf.data.experimental.AUTOTUNE)
                valid_ds = get_ds(*valid_data, BATCH_SIZE, PREFETCH_BUFFER_SIZE)
                self.model.fit(
                    train_ds, validation_data=valid_ds, epochs=EPOCHS,
                    batch_size=BATCH_SIZE)
                del train_data
                del valid_data
                del train_ds
                del valid_ds
                gc.collect()
            except StopIteration:
                print("stopped")
                break
        '''
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
