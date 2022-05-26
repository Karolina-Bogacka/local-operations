import os
import pickle
import time
from logging import INFO
from typing import Optional, Dict

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, parameters_to_weights, \
    weights_to_parameters, Config
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.grpc_server.grpc_server import start_grpc_server
from flwr.server.strategy import Strategy
from keras_preprocessing.image import ImageDataGenerator
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
from application.config import FEDERATED_PORT
from application.src.local_clients import IMAGE_SIZE, BATCH_SIZE, load_partition
from application.src.special_server import SpecialServer
from application.src.special_strategy import SpecialFedAvg

EPOCHS = 60
DEFAULT_SERVER_ADDRESS = f"[::]:8081"

def start_middle_client(config):
    client = SpecialClientImplementation(config)
    log(INFO, f"Default server address is {DEFAULT_SERVER_ADDRESS}")
    fl.client.start_numpy_client(
        server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client)



class SpecialClientImplementation(fl.client.NumPyClient):

    def __init__(self, config):
        client_manager = SimpleClientManager()
        self.config = config.dict()
        log(INFO, "after first")
        config_s = {"num_rounds": 1, "round_timeout": None}
        self.conf = {"num_rounds": 1, "round_timeout": None}
        log(INFO, "after 2nd")
        # here change the strategy in a minute to a custom one
        log(INFO, f"{self.config}")
        strategy = SpecialFedAvg(min_fit_clients=config.min_fit_clients,
                                 min_available_clients=config.min_available_clients,
                                 min_eval_clients=config.min_fit_clients)
        self.server = SpecialServer(client_manager=client_manager, strategy=strategy,
                                    timeout=int(config.timeout))
        self.index = os.getenv('USER_INDEX')
        self.losses = []
        self.grpc_server = None
        self.round = 0
        self.accuracies = []
        self.times = []
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_partition(
            int(self.index))
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
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam',
        #                   metrics=['accuracy'])
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])
        self.properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        self.num_local_rounds = (5000//BATCH_SIZE)*5
        log(INFO, f"Epochs that will be ran locally {self.num_local_rounds}")
        self.start_server(server_address=DEFAULT_SERVER_ADDRESS, config=None,
                      strategy=None)

    def start_server(
            self,
            server_address: str = DEFAULT_SERVER_ADDRESS,
            config: Optional[Dict[str, int]] = None,
            strategy: Optional[Strategy] = None
    ) -> None:

        #initialized_server, initialized_config = _init_defaults(self.server, config,
        #
        #                                                        strategy)
        log(INFO, "Before grpc server")
        self.grpc_server = start_grpc_server(
            client_manager=self.server.client_manager(),
            server_address=DEFAULT_SERVER_ADDRESS,
            max_message_length=GRPC_MAX_MESSAGE_LENGTH,
            certificates=None
        )
        log(INFO, "before conf")
        self.config["num_rounds"] = 50
        self.config["min_fit_clients"] = 1
        self.config["min_available_clients"] = 1
        log(INFO, "after conf")
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

    def get_properties(self, config):
        properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        return properties

    def get_parameters(self):  # type: ignore
        return self.model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        # Start gRPC server
        log(
            INFO,
            "Flower server running (insecure, %s rounds)",
            self.config["num_rounds"],
        )
        total_lengths = 0
        self.model.set_weights(parameters)
        new_params, lengths = self.server.fit_client_local(self.model.get_weights(),
                                                           self.round)
        total_lengths += lengths
        for i in range(self.num_local_rounds-1):
            new_params, lengths = self.server.fit_client_local(new_params, self.round)
            total_lengths += lengths
        self.round += 1
        if self.round == EPOCHS:
            self.end_server()
        log(INFO, "gathered weights")
        return new_params, total_lengths, {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model.set_weights(parameters)
        loss_aggregated, length, metrics_aggregated, (results, failures) = \
            self.server.evaluate(self.round-1)
        with open(os.path.join(os.sep, "code", "application", "results.pkl"),
                  'wb') as handle:
            self.losses.append(loss_aggregated)
            self.times.append(time.time())
            results_to_file = {"loss": self.losses,
                       "times": self.times}
            pickle.dump(results_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return loss_aggregated, length, metrics_aggregated
