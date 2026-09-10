import threading
import time
from dataclasses import dataclass, field


DEFAULT_CREDENTIAL_TTL = 2 * 60 * 60
DEFAULT_PROMPT_TTL = 24 * 60 * 60


@dataclass(frozen=True)
class Credential:
    token: str | None = field(repr=False)
    generation: int


@dataclass
class _ClientCredential:
    token: str | None = field(repr=False)
    generation: int
    updated_at: float
    connected: bool = True


@dataclass
class _PromptClient:
    client_id: str
    updated_at: float


class CredentialRegistry:
    def __init__(
        self,
        *,
        credential_ttl: float = DEFAULT_CREDENTIAL_TTL,
        prompt_ttl: float = DEFAULT_PROMPT_TTL,
        clock=time.monotonic,
    ):
        self.credential_ttl = credential_ttl
        self.prompt_ttl = prompt_ttl
        self._clock = clock
        self._lock = threading.RLock()
        self._clients: dict[str, _ClientCredential] = {}
        self._prompts: dict[str, _PromptClient] = {}

    def update(self, client_id: str, token: str | None) -> int:
        now = self._clock()
        with self._lock:
            self._cleanup(now)
            current = self._clients.get(client_id)
            if current is None:
                generation = 1
            elif current.token != token:
                generation = current.generation + 1
            else:
                current.updated_at = now
                current.connected = True
                return current.generation
            self._clients[client_id] = _ClientCredential(token, generation, now)
            return generation

    def bind_prompt(self, prompt_id: str, client_id: str) -> bool:
        now = self._clock()
        with self._lock:
            self._cleanup(now)
            if prompt_id in self._prompts:
                return False
            self._prompts[prompt_id] = _PromptClient(client_id, now)
            current = self._clients.get(client_id)
            if current is not None:
                current.connected = True
            return True

    def connect(self, client_id: str) -> None:
        with self._lock:
            current = self._clients.get(client_id)
            if current is not None:
                current.connected = True

    def get_for_prompt(self, prompt_id: str) -> Credential | None:
        now = self._clock()
        with self._lock:
            self._cleanup(now)
            prompt = self._prompts.get(prompt_id)
            if prompt is None:
                return None
            prompt.updated_at = now
            current = self._clients.get(prompt.client_id)
            if current is None:
                return None
            return Credential(current.token, current.generation)

    def release_prompt(self, prompt_id: str) -> str | None:
        with self._lock:
            prompt = self._prompts.pop(prompt_id, None)
            if prompt is None:
                return None
            current = self._clients.get(prompt.client_id)
            if current is not None and not current.connected and not self._has_prompts(prompt.client_id):
                self._clients.pop(prompt.client_id, None)
                return prompt.client_id
            if current is None and not self._has_prompts(prompt.client_id):
                return prompt.client_id
            return None

    def disconnect(self, client_id: str) -> bool:
        with self._lock:
            current = self._clients.get(client_id)
            if current is None:
                return self._has_prompts(client_id)
            current.connected = False
            if not self._has_prompts(client_id):
                self._clients.pop(client_id, None)
                return False
            return True

    def is_protected(self, client_id: str) -> bool:
        with self._lock:
            self._cleanup(self._clock())
            return client_id in self._clients or self._has_prompts(client_id)

    def clear(self) -> None:
        with self._lock:
            self._clients.clear()
            self._prompts.clear()

    def _has_prompts(self, client_id: str) -> bool:
        return any(prompt.client_id == client_id for prompt in self._prompts.values())

    def _cleanup(self, now: float) -> None:
        expired_prompts = [
            prompt_id for prompt_id, prompt in self._prompts.items() if now - prompt.updated_at >= self.prompt_ttl
        ]
        for prompt_id in expired_prompts:
            self._prompts.pop(prompt_id, None)

        active_clients = {prompt.client_id for prompt in self._prompts.values()}
        expired_clients = [
            client_id
            for client_id, credential in self._clients.items()
            if not credential.connected
            and client_id not in active_clients
            and now - credential.updated_at >= self.credential_ttl
        ]
        for client_id in expired_clients:
            self._clients.pop(client_id, None)


api_node_credentials = CredentialRegistry()
