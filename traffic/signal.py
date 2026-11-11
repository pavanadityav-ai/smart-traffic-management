"""
Adaptive signal timing and the signal state machine.

Green time = the road clearance time, kept between a minimum for the density
level and a maximum of 90 s:

    green = clamp(clearance time, MIN_GREEN_BY_DENSITY[density], MAX_GREEN_S)
    red   = max(MIN_RED_S, CYCLE_S - green - YELLOW_S)

An emergency vehicle overrides this: green immediately for 90 s.

The controller takes the current time as an argument. For a video file it is
the video time (frame number / fps), so timing does not depend on how fast the
computer processes frames; for a live camera it is the clock.
"""

import math

from traffic import config


def signal_timing(density: str, clearance_s: float, emergency: bool = False) -> dict:
    if emergency:
        return {"green_s": config.EMERGENCY_GREEN_S, "yellow_s": config.YELLOW_S, "red_s": config.EMERGENCY_RED_S,
                "reason": "Emergency vehicle priority: green to clear the path"}
    minimum = config.MIN_GREEN_BY_DENSITY[density]
    green = int(min(max(math.ceil(clearance_s), minimum), config.MAX_GREEN_S))
    red = max(config.MIN_RED_S, config.CYCLE_S - green - config.YELLOW_S)
    if green == config.MAX_GREEN_S and clearance_s > config.MAX_GREEN_S:
        reason = f"{density} density: clearance {clearance_s:.0f} s, capped at {config.MAX_GREEN_S} s"
    elif green > minimum:
        reason = f"{density} density: green set to the clearance time ({clearance_s:.0f} s)"
    else:
        reason = f"{density} density: minimum green for this level ({minimum} s)"
    return {"green_s": green, "yellow_s": config.YELLOW_S, "red_s": red, "reason": reason}


class SignalController:
    """GREEN -> YELLOW -> RED -> GREEN ...

    * A running phase is never cut short by a density change; new timing is
      used when the next phase starts.
    * An emergency switches to GREEN immediately, once per emergency event.
    """

    def __init__(self, start_time: float = 0.0) -> None:
        self.phase = "RED"
        self.phase_start = start_time
        self.now = start_time
        self.green_s = config.MIN_GREEN_BY_DENSITY["Low"]
        self.red_s = config.MIN_RED_S
        self.reason = "Starting"
        self._emergency_active = False

    def remaining(self, now: float | None = None) -> float:
        now = self.now if now is None else now
        duration = {"GREEN": self.green_s, "YELLOW": config.YELLOW_S, "RED": self.red_s}[self.phase]
        return max(0.0, duration - (now - self.phase_start))

    def update(self, timing: dict, emergency: bool, now: float) -> None:
        self.now = now
        elapsed = now - self.phase_start
        if emergency and not self._emergency_active:
            self._emergency_active = True
            self._enter("GREEN", now, timing)
            return
        if not emergency:
            self._emergency_active = False
        if self.phase == "GREEN" and elapsed >= self.green_s:
            self._enter("YELLOW", now, timing)
        elif self.phase == "YELLOW" and elapsed >= config.YELLOW_S:
            self._enter("RED", now, timing)
        elif self.phase == "RED" and elapsed >= self.red_s:
            self._enter("GREEN", now, timing)

    def _enter(self, phase: str, now: float, timing: dict) -> None:
        self.phase, self.phase_start = phase, now
        if phase == "GREEN":
            self.green_s = timing["green_s"]
            self.reason = timing["reason"]
        elif phase == "RED":
            self.red_s = timing["red_s"]

    def state(self) -> dict:
        return {"phase": self.phase, "remaining_s": round(self.remaining(), 1),
                "green_s": self.green_s, "red_s": self.red_s, "yellow_s": config.YELLOW_S, "reason": self.reason}
