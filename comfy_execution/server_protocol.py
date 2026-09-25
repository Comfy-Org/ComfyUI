from typing import Protocol


class ExecutionServer(Protocol):
    client_id: str | None
    last_node_id: str | None
    sockets_metadata: dict[str, dict[str, object]]

    def send_sync(self, event: str | int, data: object, sid: str | None = None) -> None: ...

    def queue_updated(self) -> None: ...
