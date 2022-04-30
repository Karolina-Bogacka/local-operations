"""Flower client app."""


import time
from logging import INFO
from typing import Optional, Union

from flwr.client import Client
from flwr.client.grpc_client.connection import grpc_connection
from flwr.client.grpc_client.message_handler import handle
from .my_client import has_get_properties as myclient_has_get_properties
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log

from application.my_client import MyClient, MyClientWrapper

ClientLike = Union[Client, MyClient]


def start_client(
    server_address: str,
    client: Client,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower Client which connects to a gRPC server.
    Parameters
    ----------
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        client: flwr.client.Client. An implementation of the abstract base
            class `flwr.client.Client`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.server.start_server`), otherwise it will not
            know about the increased limit and block larger messages.
        root_certificates: bytes (default: None)
            The PEM-encoded root certificates as a byte string. If provided, a secure
            connection using the certificates will be established to a
            SSL-enabled Flower server.
    Returns
    -------
        None
    Examples
    --------
    Starting a client with insecure server connection:

    """
    while True:
        sleep_duration: int = 0
        with grpc_connection(
            server_address,
            max_message_length=grpc_max_message_length,
            root_certificates=root_certificates,
        ) as conn:
            receive, send = conn

            while True:
                server_message = receive()
                client_message, sleep_duration, keep_going = handle(
                    client, server_message
                )
                send(client_message)
                if not keep_going:
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


def start_my_client(
    server_address: str,
    client: MyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.
    Parameters
    ----------
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        client: flwr.client.NumPyClient. An implementation of the abstract base
            class `flwr.client.NumPyClient`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.server.start_server`), otherwise it will not
            know about the increased limit and block larger messages.
        root_certificates: bytes (default: None)
            The PEM-encoded root certificates a byte string. If provided, a secure
            connection using the certificates will be established to a
            SSL-enabled Flower server.
    Returns
    -------
        None
    Examples
    --------
    """

    # Wrap the NumPyClient
    flower_client = MyClientWrapper(client)

    # Delete get_properties method from NumPyClientWrapper if the user-provided
    # NumPyClient instance does not implement get_properties. This enables the
    # following call to start_client to handle NumPyClientWrapper instances like any
    # other Client instance (which might or might not implement get_properties).
    if not myclient_has_get_properties(client=client):
        del MyClientWrapper.get_properties

    # Start
    start_client(
        server_address=server_address,
        client=flower_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
    )


def to_client(client_like: ClientLike) -> Client:
    """Take any Client-like object and return it as a Client."""
    if isinstance(client_like, MyClient):
        return MyClientWrapper(numpy_client=client_like)
    return client_like
