"""
Async semaphore and concurrency control

Provides semaphores for limiting concurrent operations and preventing
resource exhaustion in async environments.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SemaphoreStats:
    """Statistics for semaphore usage"""

    name: str
    capacity: int
    current_value: int
    waiting_count: int
    total_acquisitions: int
    total_timeouts: int
    average_hold_time: float
    max_hold_time: float

    @property
    def utilization(self) -> float:
        """Current utilization as percentage"""
        return ((self.capacity - self.current_value) / self.capacity) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "current_value": self.current_value,
            "waiting_count": self.waiting_count,
            "utilization_percent": self.utilization,
            "total_acquisitions": self.total_acquisitions,
            "total_timeouts": self.total_timeouts,
            "average_hold_time": self.average_hold_time,
            "max_hold_time": self.max_hold_time,
        }


class AsyncSemaphore:
    """
    Enhanced async semaphore with statistics and timeout support

    Features:
    - Timeout support for acquisitions
    - Detailed usage statistics
    - Fair queuing (FIFO)
    - Resource leak detection
    """

    def __init__(self, value: int, name: str = "unnamed"):
        """
        Initialize async semaphore

        Args:
            value: Initial semaphore value (capacity)
            name: Name for identification and logging
        """
        if value < 0:
            raise ValueError("Semaphore value must be non-negative")

        self.name = name
        self.capacity = value
        self._semaphore = asyncio.Semaphore(value)

        self._total_acquisitions = 0
        self._total_timeouts = 0
        self._hold_times = []
        self._max_hold_time = 0.0

        self._active_acquisitions = {}
        self._next_acquisition_id = 0

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire semaphore with optional timeout

        Args:
            timeout: Maximum time to wait for acquisition (None = no timeout)

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        acquisition_id = self._next_acquisition_id
        self._next_acquisition_id += 1


        try:
            if timeout is not None:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            else:
                await self._semaphore.acquire()

            acquire_time = time.time()
            self._active_acquisitions[acquisition_id] = acquire_time
            self._total_acquisitions += 1

            logger.debug(f"Semaphore '{self.name}' acquired (id={acquisition_id})")

            try:
                yield
            finally:
                if acquisition_id in self._active_acquisitions:
                    hold_time = time.time() - self._active_acquisitions[acquisition_id]
                    self._hold_times.append(hold_time)
                    self._max_hold_time = max(self._max_hold_time, hold_time)
                    del self._active_acquisitions[acquisition_id]

                self._semaphore.release()
                logger.debug(f"Semaphore '{self.name}' released (id={acquisition_id})")

        except asyncio.TimeoutError:
            self._total_timeouts += 1
            logger.warning(
                f"Semaphore '{self.name}' acquisition timeout after {timeout}s"
            )
            raise

    def locked(self) -> bool:
        """Check if semaphore is at capacity (no permits available)"""
        # Using private member is necessary for semaphore inspection
        return self._semaphore._value == 0  # noqa: SLF001

    def get_stats(self) -> SemaphoreStats:
        """Get detailed semaphore statistics"""
        current_value = self._semaphore._value  # noqa: SLF001
        waiting_count = (
            len(self._semaphore._waiters)
            if hasattr(self._semaphore, "_waiters")
            else 0  # noqa: SLF001
        )

        avg_hold_time = (
            sum(self._hold_times) / len(self._hold_times) if self._hold_times else 0.0
        )

        return SemaphoreStats(
            name=self.name,
            capacity=self.capacity,
            current_value=current_value,
            waiting_count=waiting_count,
            total_acquisitions=self._total_acquisitions,
            total_timeouts=self._total_timeouts,
            average_hold_time=avg_hold_time,
            max_hold_time=self._max_hold_time,
        )

    def check_leaks(self, max_hold_time: float = 300.0) -> int:
        """
        Check for potential resource leaks

        Args:
            max_hold_time: Maximum reasonable hold time in seconds

        Returns:
            Number of potentially leaked acquisitions
        """
        current_time = time.time()
        leaked_count = 0

        for acquisition_id, acquire_time in list(self._active_acquisitions.items()):
            hold_time = current_time - acquire_time
            if hold_time > max_hold_time:
                logger.warning(
                    f"Potential semaphore leak in '{self.name}': "
                    f"acquisition {acquisition_id} held for {hold_time:.1f}s"
                )
                leaked_count += 1

        return leaked_count


class SemaphoreManager:
    """
    Centralized manager for named semaphores

    Provides creation, management, and monitoring of semaphores
    across the application.
    """

    def __init__(self):
        self._semaphores: dict[str, AsyncSemaphore] = {}
        self._default_capacities = {
            "llm_requests": 10,
            "memory_operations": 20,
            "extractor_operations": 15,
            "rca_pipeline": 5,
            "action_pipeline": 5,
        }

    def get_semaphore(
        self, name: str, capacity: Optional[int] = None
    ) -> AsyncSemaphore:
        """
        Get or create a named semaphore

        Args:
            name: Semaphore name
            capacity: Semaphore capacity (uses default if None)

        Returns:
            AsyncSemaphore instance
        """
        if name not in self._semaphores:
            if capacity is None:
                capacity = self._default_capacities.get(name, 10)

            self._semaphores[name] = AsyncSemaphore(capacity, name)
            logger.info(f"Created semaphore '{name}' with capacity {capacity}")

        return self._semaphores[name]

    def list_semaphores(self) -> dict[str, SemaphoreStats]:
        """Get statistics for all semaphores"""
        return {
            name: semaphore.get_stats() for name, semaphore in self._semaphores.items()
        }

    def check_all_leaks(self, max_hold_time: float = 300.0) -> dict[str, int]:
        """Check all semaphores for potential leaks"""
        leak_counts = {}

        for name, semaphore in self._semaphores.items():
            leak_count = semaphore.check_leaks(max_hold_time)
            if leak_count > 0:
                leak_counts[name] = leak_count

        return leak_counts

    def set_default_capacity(self, name: str, capacity: int) -> None:
        """Set default capacity for a semaphore type"""
        self._default_capacities[name] = capacity

    async def acquire_multiple(
        self, semaphore_names: list[str], timeout: Optional[float] = None
    ) -> list[AsyncSemaphore]:
        """
        Acquire multiple semaphores atomically

        Args:
            semaphore_names: List of semaphore names to acquire
            timeout: Total timeout for all acquisitions

        Returns:
            List of acquired semaphores

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        semaphores = [self.get_semaphore(name) for name in semaphore_names]

        sorted_semaphores = sorted(semaphores, key=lambda s: s.capacity)

        acquired = []

        try:
            for semaphore in sorted_semaphores:
                remaining_timeout = timeout
                if timeout is not None and acquired:
                    elapsed = sum(
                        time.time()
                        - s._active_acquisitions.get(0, time.time())  # noqa: SLF001
                        for s in acquired
                    )
                    remaining_timeout = max(0, timeout - elapsed)

                await semaphore.acquire(remaining_timeout).__aenter__()
                acquired.append(semaphore)

            return acquired

        except Exception:
            for semaphore in reversed(acquired):
                try:
                    semaphore._semaphore.release()  # noqa: SLF001
                except Exception as e:
                    logger.error(f"Error releasing semaphore during cleanup: {e}")
            raise


_semaphore_manager = SemaphoreManager()


def get_semaphore_manager() -> SemaphoreManager:
    """Get the global semaphore manager"""
    return _semaphore_manager


def limit_concurrency(
    semaphore_name: str, capacity: Optional[int] = None, timeout: Optional[float] = None
):
    """
    Decorator to limit function concurrency using semaphores

    Args:
        semaphore_name: Name of semaphore to use
        capacity: Semaphore capacity (None uses default)
        timeout: Acquisition timeout
    """

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            manager = get_semaphore_manager()
            semaphore = manager.get_semaphore(semaphore_name, capacity)

            async with semaphore.acquire(timeout):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        logger.warning(
            f"Concurrency limiting not supported for sync function {func.__name__}"
        )
        return sync_wrapper

    return decorator
