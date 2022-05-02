import os
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

from application.config import FEDERATED_PORT
from application.src.local_clients import get_cnn_model_1, IMAGE_SIZE
from application.src.special_server import SpecialServer
from application.src.special_strategy import SpecialFedAvg


def start_middle_client(config):
    client = SpecialClientImplementation(config)
    fl.client.start_numpy_client(
        server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client)


DEFAULT_SERVER_ADDRESS = "[::]:8081"


class SpecialClientImplementation(fl.client.NumPyClient):

    def __init__(self, config):
        client_manager = SimpleClientManager()
        self.config = config.dict()
        log(INFO, "after first")
        config_s = {"num_rounds": 1, "round_timeout": None}
        self.conf = {"num_rounds": 1, "round_timeout": None}
        log(INFO, "after 2nd")
        # here change the strategy in a minute to a custom one
        strategy = SpecialFedAvg()
        self.server = SpecialServer(client_manager=client_manager, strategy=strategy)
        self.index = os.getenv('USER_INDEX')
        self.losses = []
        self.grpc_server = None
        self.round = 0
        self.accuracies = []
        self.precisions = []
        self.properties = {"GROUP_INDEX": os.environ["GROUP_INDEX"]}
        all_training = os.walk(
            os.path.join(os.sep, 'data', 'new_split', f'fold_{self.index}'))
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
        self.num_local_rounds = 600
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
        log(INFO, f"{type(parameters[0])}")
        log(INFO, f"{type(weights_to_parameters(parameters).tensors[0])}")
        new_params, lengths = self.server.fit_client_local(self.model.get_weights(),
                                                           self.round)
        total_lengths += lengths
        for i in range(self.num_local_rounds-1):
            new_params, lengths = self.server.fit_client_local(new_params, self.round)
            total_lengths += lengths
        self.round += 1
        if self.round == config["num_rounds"]:
            self.end_server()
        log(INFO, "gathered weights")
        return new_params, total_lengths, {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model.set_weights(parameters)
        losses, lengths, metrics = 0.0, 10, {m: 0.0 for m in self.metric_names}
        if self.round == config["num_rounds"]:
            self.end_server()
        return losses, lengths, metrics
