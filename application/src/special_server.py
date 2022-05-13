import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import List, Optional, Tuple, Dict

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Reconnect,
    Weights,
    parameters_to_weights, weights_to_parameters, Parameters, Scalar,
)
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from application.src.special_strategy import SpecialFedAvg

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]


def set_strategy(strategy: Optional[Strategy]) -> Strategy:
    """Return Strategy."""
    return strategy if strategy is not None else SpecialFedAvg()


class SpecialServer(Server):
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None,
            timeout: int = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.weights: Weights = []
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = set_strategy(strategy)
        self.timeout = timeout
        self.max_workers: Optional[int] = None

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def fit_client_local(self, weights, current_round):
        self.weights = weights
        log(INFO, "Middle level FL starting")
        log(
            DEBUG,
            "before fit round: strategy has %s clients",
            self._client_manager.num_available(),
        )
        res_fit = self.fit_round(rnd=current_round)
        if res_fit:
            parameters_aggregated, metrics_aggregated, lengths, (results, failures) = res_fit
            if parameters_aggregated:
                self.weights = parameters_to_weights(parameters_aggregated)
                self.parameters = parameters_aggregated
        return self.weights, lengths


    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.weights = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=weights_to_parameters(self.weights))
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(rnd=current_round)
            log(INFO, f"Evaluate results look like {res_fed}")
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        rnd=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, metrics, length, results_and_failures = res
        return loss, length, metrics, results_and_failures

    def evaluate_round(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=weights_to_parameters(self.weights),
            client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=None,
            timeout=None,
        )
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result = self.strategy.aggregate_evaluate(rnd, results, failures)
        log(INFO, f"Evaluate results look like {aggregated_result}")
        loss_aggregated, length, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, length, (results, failures)

    def fit_round(
        self,
        rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=weights_to_parameters(self.weights),
            client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=None,
            timeout=self.timeout,
        )
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_fit(rnd, results, failures)

        parameters_aggregated, metrics_aggregated, lengths = aggregated_result
        return parameters_aggregated, metrics_aggregated, lengths, (results, failures)

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

    def _get_initial_parameters(self, timeout=None) -> Weights:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Weights] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Received initial parameters from strategy")
            return parameters

        # Get initial parameters from one of the clients
        random_client = self._client_manager.sample(1)[0]
        log(INFO, f"This client's cid is {random_client.cid} with properties "
                  f"{random_client.properties} of bridge {random_client.bridge} active "
                  f"with "
                  f"{random_client.bridge._is_closed()}")
        parameters_res = random_client.get_parameters()
        log(INFO, f"This client's cid is {random_client.cid} with properties "
                  f"{random_client.properties} of bridge {random_client.bridge} active "
                  f"with "
                  f"{random_client.bridge._is_closed()}")
        weights = parameters_to_weights(parameters_res.parameters)
        log(INFO, "Received initial parameters from one random client")
        return weights


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float]
) -> FitResultsAndFailures:
    """Refine weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        log(INFO, f"{client_instructions[0][0].cid}")
        log(INFO, f"{client_instructions[0][1].parameters.tensor_type}")
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=timeout,  # Handled in the respective communication stack
        )

        # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        log(INFO, f"Exception of nature: {failure}")
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res



def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int]=None,
    timeout: Optional[float]=None,
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate weights on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
