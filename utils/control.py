"""
Signal control logic.

Provides:
  - calculate_signal_timing()   timing rules for each density level / emergency
  - get_signal_recommendation() one-shot recommendation from a vehicle count
  - TrafficSignalController     a GREEN -> YELLOW -> RED state machine

The controller takes the current time as an argument (`now`). In the live
detection page `now` is the *video* time (frame number / frames per second),
so the signal behaves the same on a fast or a slow computer. In the simulator
it is the real clock. Passing time in also makes the controller easy to test.
"""

from utils.density import classify_density

YELLOW_DURATION = 5  # seconds, always fixed

# When this lane is RED, cross-traffic is moving. If our lane is busy we keep
# the red short; if our lane is quiet, cross-traffic gets a long green.
#
# Condition        Green   Red   Reason
# Emergency          90     10   Clear the corridor, minimal red wait
# High density       60     25   Move the queue; short cross-phase
# Medium density     40     55   Balanced exchange
# Low density        20     95   Little traffic; cross-traffic has priority
_TIMING = {
    "emergency": {
        "green_time": 90,
        "red_time": 10,
        "green_status": "🚨 Emergency override: clearing the path",
        "red_status": "🚨 Emergency nearby: holding cross-traffic to 10 s",
    },
    "High": {
        "green_time": 60,
        "red_time": 25,
        "green_status": "⚠️ Heavy traffic: extended green phase (60 s)",
        "red_status": "⚠️ Heavy traffic here: short red wait (25 s)",
    },
    "Medium": {
        "green_time": 40,
        "red_time": 55,
        "green_status": "🟡 Moderate traffic: standard green phase (40 s)",
        "red_status": "🟡 Moderate flow: balanced red wait (55 s)",
    },
    "Low": {
        "green_time": 20,
        "red_time": 95,
        "green_status": "✅ Light traffic: short green phase (20 s)",
        "red_status": "✅ Light flow: cross-traffic priority, long red (95 s)",
    },
}


def calculate_signal_timing(density: str, emergency_detected: bool = False) -> dict:
    """Timing for a density level. Keys: green_time, yellow_time, red_time, green_status, red_status."""
    row = _TIMING["emergency" if emergency_detected else density]
    return {
        "green_time": row["green_time"],
        "yellow_time": YELLOW_DURATION,
        "red_time": row["red_time"],
        "green_status": row["green_status"],
        "red_status": row["red_status"],
    }


def get_signal_recommendation(vehicle_count: int, emergency_detected: bool = False,
                              low_threshold: int = 10, high_threshold: int = 25) -> dict:
    """One-shot recommendation from a vehicle count (adds density, vehicle_count, emergency)."""
    density = classify_density(vehicle_count, low_threshold, high_threshold)
    return {
        "density": density,
        "vehicle_count": vehicle_count,
        "emergency": emergency_detected,
        **calculate_signal_timing(density, emergency_detected),
    }


class TrafficSignalController:
    """Traffic-signal state machine: GREEN -> YELLOW -> RED -> GREEN ...

    * A phase is never cut short by a density change; the new timing is used
      when the next phase starts.
    * An emergency switches to GREEN (90 s) immediately, once per emergency event.
    * The RED and GREEN durations are fixed at the moment each phase starts.
    """

    def __init__(self, start_time: float = 0.0) -> None:
        self.phase = "RED"  # start on RED so the first GREEN uses fresh timing
        self._phase_start = start_time
        self._now = start_time
        self.green_duration = _TIMING["Low"]["green_time"]
        self.red_duration = _TIMING["Low"]["red_time"]
        self.green_status = ""
        self.red_status = ""
        self._emergency_active = False

    # ---- read-only information -------------------------------------------------
    def remaining(self, now: float | None = None) -> float:
        """Seconds left in the current phase (never negative)."""
        now = self._now if now is None else now
        duration = {"GREEN": self.green_duration, "YELLOW": YELLOW_DURATION, "RED": self.red_duration}[self.phase]
        return max(0.0, duration - (now - self._phase_start))

    @property
    def current_status(self) -> str:
        return self.red_status if self.phase == "RED" else self.green_status

    # ---- update ----------------------------------------------------------------
    def update(self, signal: dict, now: float) -> None:
        """Advance the state machine. Call once per processed frame with the current time."""
        self._now = now
        elapsed = now - self._phase_start
        emergency = signal.get("emergency", False)

        if emergency and not self._emergency_active:
            self._transition_to("GREEN", now, signal)  # cut straight to GREEN
            self._emergency_active = True
            return
        if not emergency:
            self._emergency_active = False

        if self.phase == "GREEN" and elapsed >= self.green_duration:
            self._transition_to("YELLOW", now, signal)
        elif self.phase == "YELLOW" and elapsed >= YELLOW_DURATION:
            self._transition_to("RED", now, signal)
        elif self.phase == "RED" and elapsed >= self.red_duration:
            self._transition_to("GREEN", now, signal)

    def _transition_to(self, new_phase: str, now: float, signal: dict) -> None:
        self.phase = new_phase
        self._phase_start = now
        self.green_status = signal.get("green_status", "")
        self.red_status = signal.get("red_status", "")
        if new_phase == "GREEN":
            self.green_duration = signal["green_time"]
        elif new_phase == "RED":
            self.red_duration = signal["red_time"]
