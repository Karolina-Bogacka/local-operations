import os
import pickle
import threading
import time
from logging import INFO
from typing import Optional, Dict

import flwr as fl
import numpy as np
from flwr.client import start_numpy_client
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.grpc_server.grpc_server import start_grpc_server
from flwr.server.strategy import Strategy
from keras import backend, Sequential
from keras.constraints import maxnorm
from keras.datasets.cifar import load_batch
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from application.src.local_clients import BATCH_SIZE
from application.src.decentralized_fedavg import DecentralizedFedAvg
from application.src.decentralized_server import DecentralizedServer
from application.src.small_client import SmallCifarClient

EPOCHS=5

current_jobs = {}
ERAS = 1
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


def start_big_client(id, config):
    log(INFO, f"Connect big client to main server {config.server_address}:8080")
    client = BigCifarClient(config)
    start_numpy_client(server_address=f"{config.server_address}:8080",
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
               x_train[idx * 8333: (idx + 1) * 8333],
               y_train[idx * 8333: (idx + 1) * 8333],
           ), (
               x_test[idx * 1666: (idx + 1) * 1666],
               y_test[idx * 1666: (idx + 1) * 1666],
           )

DEFAULT_SERVER_ADDRESS = f"[::]:8080"

class BigCifarClient(fl.client.NumPyClient):

    def __init__(self, config):
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
        self.current_round = 0
        self.current_step = 0
        self.step_diff = 1
        self.possible_steps = len(self.x_train) // (BATCH_SIZE * self.step_diff)
        self.rounds = 5
        # start server and client manager
        client_manager = SimpleClientManager()
        self.config = config.dict()
        log(INFO, "after first")
        config_s = {"num_rounds": 1, "round_timeout": None}
        self.conf = {"num_rounds": 1, "round_timeout": None}
        # here change the strategy in a minute to a custom one
        strategy = DecentralizedFedAvg()
        self.server = DecentralizedServer(client_manager=client_manager,
                                          strategy=strategy)
        self.start_server()


    def start_server(
            self,
            server_address: str = DEFAULT_SERVER_ADDRESS,
            config: Optional[Dict[str, int]] = None,
            strategy: Optional[Strategy] = None
    ) -> None:

        #initialized_server, initialized_config = _init_defaults(self.server, config,
        #
        #                                                        strategy)
        self.grpc_server = start_grpc_server(
            client_manager=self.server.client_manager(),
            server_address=DEFAULT_SERVER_ADDRESS,
            max_message_length=GRPC_MAX_MESSAGE_LENGTH,
            certificates=None
        )
        self.config["num_rounds"] = 5
        self.config["min_fit_clients"] = 1
        self.config["min_available_clients"] = 1
        log(
            INFO,
            "Flower server running (insecure, %s rounds)",
            self.config["num_rounds"],
        )
        # Start gRPC server
        log(INFO, "Before grpc server")
        log(
            INFO,
            "Flower server running (insecure, %s rounds)",
            self.config["num_rounds"],
        )

    def end_server(self):
        # Stop the gRPC server
        self.grpc_server.stop(grace=1)

    def start_small_client(self, config):
        e = threading.Event()
        self.client = SmallCifarClient(e)
        t1 = threading.Thread(name='small client thread',
                              target=start_numpy_client,
                              args=(f"{config.server_address}:8080", self.client))
        t1.start()

    def get_parameters(self):
        return self.model.get_weights()

    def get_properties(self, config):
        properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        return properties

    def fit(self, parameters, config):
        log(INFO, f"Starting training for n rounds {len(parameters)}")
        self.model.set_weights(parameters)
        for n in range(self.rounds):
            log(INFO, f"Fit in round {n}")
            self.weights, self.lengths = self.server.fit_client_local(
                self.model.get_weights(), n)
            log(INFO, f"Stuck waiting {self.client.flag.is_set()}")
            self.client.flag.wait()
            self.model.set_weights(self.client.model.get_weights())
            self.client.flag.clear()
            log(INFO, f"Cleared flag to {self.client.flag.is_set()}")
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
