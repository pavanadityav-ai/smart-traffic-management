"""
Traffic density classification.

Turns a vehicle count into a density level (Low / Medium / High) and smooths
the count over recent frames so the level does not flicker when the detector
misses or adds a vehicle in a single frame.
"""

from collections import deque
from statistics import median


def classify_density(vehicle_count: int, low_threshold: int = 10, high_threshold: int = 25) -> str:
    """Classify traffic density from a vehicle count.

    Low:    count <  low_threshold
    Medium: low_threshold <= count < high_threshold
    High:   count >= high_threshold
    """
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must not be greater than high_threshold")
    if vehicle_count < low_threshold:
        return "Low"
    if vehicle_count < high_threshold:
        return "Medium"
    return "High"


class CountSmoother:
    """Median of the last `window` vehicle counts.

    A single frame can miss a car hidden behind a bus, or count one car twice.
    The median of recent frames ignores these one-frame spikes, so the density
    level (and therefore the signal timing) is more stable.
    """

    def __init__(self, window: int = 15) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self._counts: deque = deque(maxlen=window)

    def update(self, count: int) -> int:
        """Add the newest count and return the smoothed count."""
        self._counts.append(count)
        return int(round(median(self._counts)))

    def reset(self) -> None:
        self._counts.clear()


def get_density_color(density: str) -> str:
    """CSS colour for a density level."""
    return {"Low": "green", "Medium": "orange", "High": "red"}.get(density, "gray")


def get_density_emoji(density: str) -> str:
    """Coloured-circle indicator for a density level."""
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(density, "⚪")
