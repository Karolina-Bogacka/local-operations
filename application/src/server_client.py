"""Flower client app."""


import timeit
from abc import ABC, abstractmethod
from logging import  INFO
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    ParametersRes,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.client import Client
from flwr.common.logger import log

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:
    Tuple[float, int, Dict[str, Scalar]]
Example
-------
    0.5, 10, {"accuracy": 0.95}
"""


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters.
        Returns:
            The local model parameters as a list of NumPy ndarrays.
        """

    @abstractmethod
    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.
        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.
        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    @abstractmethod
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],  # Deprecated
        Tuple[int, float, float, Dict[str, Scalar]],  # Deprecated
    ]:
        """Evaluate the provided weights using the locally held dataset.
        Args:
            parameters (List[np.ndarray]): The current (global) model
                parameters.
            config (Dict[str, Scalar]): Configuration parameters which allow the
                server to influence evaluation on the client. It can be used to
                communicate arbitrary values from the server to the client, for
                example, to influence the number of examples used for
                evaluation.
        Returns:
            loss (float): The evaluation loss of the model on the local
                dataset.
            num_examples (int): The number of examples used for evaluation.
            metrics (Dict[str, Scalar]): A dictionary mapping arbitrary string
                keys to values of type bool, bytes, float, int, or str. It can
                be used to communicate arbitrary values back to the server.
        Notes:
            The previous return type format (int, float, float) and the
            extended format (int, float, float, Dict[str, Scalar]) are still
            supported for compatibility reasons. They will however be removed
            in a future release, please migrate to
            (float, int, Dict[str, Scalar]).
        """


class NumPyClientWrapper(Client):
    """Wrapper which translates between Client and NumPyClient."""

    def __init__(self, numpy_client: NumPyClient) -> None:
        self.numpy_client = numpy_client

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        parameters = self.numpy_client.get_parameters()
        parameters_proto = weights_to_parameters(parameters)
        return ParametersRes(parameters=parameters_proto)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        log(INFO, "training reached")
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)
        results: Tuple[List[np.ndarray], int, Metrics] = self.numpy_client.fit(
            parameters, ins.config
        )
        # Train
        fit_begin = timeit.default_timer()
        if not (
                len(results) == 3
                and isinstance(results[0], list)
                and isinstance(results[1], int)
                and isinstance(results[2], dict)
        ):
            raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

            # Return FitRes
        fit_duration = timeit.default_timer() - fit_begin
        parameters_prime, num_examples, metrics = results
        parameters_prime_proto = weights_to_parameters(parameters_prime)
        return FitRes(
            parameters=parameters_prime_proto,
            num_examples=num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        results = self.numpy_client.evaluate(parameters, ins.config)
        if len(results) == 3:
            if (
                isinstance(results[0], float)
                and isinstance(results[1], int)
                and isinstance(results[2], dict)
            ):
                # Forward-compatible case: loss, num_examples, metrics
                results = cast(Tuple[float, int, Metrics], results)
                loss, num_examples, metrics = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    metrics=metrics,
                )
            elif (
                isinstance(results[0], int)
                and isinstance(results[1], float)
                and isinstance(results[2], float)
            ):
                # Legacy case: num_examples, loss, accuracy
                # This will be removed in a future release
                results = cast(Tuple[int, float, float], results)
                num_examples, loss, accuracy = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    accuracy=accuracy,  # Deprecated
                )
            else:
                raise Exception(
                    "Return value expected to be of type (float, int, dict)."
                )
        elif len(results) == 4:
            # Legacy case: num_examples, loss, accuracy, metrics
            # This will be removed in a future release
            results = cast(Tuple[int, float, float, Metrics], results)
            assert isinstance(results[0], int)
            assert isinstance(results[1], float)
            assert isinstance(results[2], float)
            assert isinstance(results[3], dict)
            num_examples, loss, accuracy, metrics = results
            evaluate_res = EvaluateRes(
                loss=loss,
                num_examples=num_examples,
                accuracy=accuracy,  # Deprecated
                metrics=metrics,
            )
        return evaluate_res