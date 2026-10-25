"""Per-model circuit breaker.

After `failure_threshold` consecutive provider-side failures a model is skipped
for `cooldown_s` seconds, so a provider outage costs one failed call per
cooldown window instead of one per request. After the cooldown one trial call
is allowed through (half-open); success closes the circuit again.

State is in-process memory, i.e. per API worker.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _State:
    failures: int = 0
    opened_at: float | None = None


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_s: float = 60.0
    clock: Callable[[], float] = time.monotonic  # injectable for tests
    _states: dict[int, _State] = field(default_factory=dict)

    def allow(self, model_id: int) -> bool:
        s = self._states.get(model_id)
        if s is None or s.opened_at is None:
            return True
        if self.clock() - s.opened_at >= self.cooldown_s:
            s.opened_at = self.clock()  # half-open: let one trial through, re-arm timer
            return True
        return False

    def record_success(self, model_id: int) -> None:
        self._states.pop(model_id, None)

    def record_failure(self, model_id: int) -> None:
        s = self._states.setdefault(model_id, _State())
        s.failures += 1
        if s.failures >= self.failure_threshold:
            s.opened_at = self.clock()

    def is_open(self, model_id: int) -> bool:
        s = self._states.get(model_id)
        return s is not None and s.opened_at is not None


default_breaker = CircuitBreaker()
