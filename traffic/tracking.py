"""
A simple IoU tracker. It gives each vehicle an ID across frames, which lets us:

  * count each vehicle once (unique count), not once per frame;
  * handle occlusion: a vehicle hidden behind a bus for a few frames keeps its
    track and is still counted in the queue (up to TRACK_MAX_MISSES frames);
  * measure speed from how far the vehicle moves between frames.

Matching is greedy on IoU (intersection over union of the boxes), which works
well at video frame rates where boxes move only a little between frames.
"""

from collections import deque
from itertools import count

from traffic.config import TRACK_IOU_MATCH, TRACK_MAX_MISSES, TRACK_MIN_HITS


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id: int, det, t: float) -> None:
        self.id = track_id
        self.box = det.box
        self.label = det.label
        self.emergency = det.emergency
        self.confidence = det.confidence
        self.hits = 1
        self.misses = 0
        self.history = deque([(t, *self.centre)], maxlen=15)

    @property
    def centre(self) -> tuple:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def confirmed(self) -> bool:
        return self.hits >= TRACK_MIN_HITS

    def update(self, det, t: float) -> None:
        self.box, self.label, self.confidence = det.box, det.label, det.confidence
        self.emergency = det.emergency  # follows the latest label, so an ID mix-up does not stick
        self.hits += 1
        self.misses = 0
        self.history.append((t, *self.centre))

    def speed_px_per_s(self) -> float | None:
        """Average speed over the recent history, in pixels per second."""
        if len(self.history) < 3:
            return None
        (t0, x0, y0), (t1, x1, y1) = self.history[0], self.history[-1]
        if t1 <= t0:
            return None
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (t1 - t0)


class Tracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self._ids = count(1)
        self.unique_counts: dict[str, int] = {}

    def update(self, detections: list, t: float) -> list[Track]:
        """Match detections to tracks. Returns the confirmed tracks (visible or briefly hidden)."""
        pairs = sorted(((iou(tr.box, d.box), i, j) for i, tr in enumerate(self.tracks) for j, d in enumerate(detections)),
                       reverse=True)
        used_tracks, used_dets = set(), set()
        for score, i, j in pairs:
            if score < TRACK_IOU_MATCH:
                break
            if i in used_tracks or j in used_dets:
                continue
            used_tracks.add(i)
            used_dets.add(j)
            track = self.tracks[i]
            was_confirmed = track.confirmed
            track.update(detections[j], t)
            if track.confirmed and not was_confirmed:
                self.unique_counts[track.label] = self.unique_counts.get(track.label, 0) + 1

        for i, track in enumerate(self.tracks):
            if i not in used_tracks:
                track.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= TRACK_MAX_MISSES]

        for j, det in enumerate(detections):
            if j not in used_dets:
                track = Track(next(self._ids), det, t)
                self.tracks.append(track)
                if track.confirmed:  # only when TRACK_MIN_HITS is 1
                    self.unique_counts[track.label] = self.unique_counts.get(track.label, 0) + 1

        return [tr for tr in self.tracks if tr.confirmed]
