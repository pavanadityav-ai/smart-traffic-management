"""
Edge-case handling before detection.

  Low light            mean brightness is low   -> flagged; optional CLAHE + gamma brightening
  Fog, rain, haze      grey-level contrast low  -> flagged; optional CLAHE
                       (the enhancement is off by default: the benchmark showed it
                       lowered YOLOv8 accuracy on darkened images)
  Camera shake         whole frame shifts between frames -> shift is smoothed out
                       (points are tracked with optical flow; if many of them stay still, only
                       vehicles moved; otherwise RANSAC finds the shift shared by most points)
  Blurry frame         Laplacian variance low   -> flagged (detections less reliable)
  Frozen camera        the exact same frame repeats -> flagged after FROZEN_FRAMES frames

Occlusion is handled later, by the tracker (traffic/tracking.py).
Every frame gets a `flags` dict, so the dashboard and the database show when
the conditions were difficult.
"""

import cv2
import numpy as np

from traffic import config

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_GAMMA_LUT = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)  # brightens dark areas


def enhance_contrast(frame: np.ndarray, brighten: bool = False) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = _CLAHE.apply(l_chan)
    if brighten:
        l_chan = cv2.LUT(l_chan, _GAMMA_LUT)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def global_shift(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Shift (dx, dy) of the whole image between two grey frames, robust to moving vehicles."""
    points = cv2.goodFeaturesToTrack(prev.astype(np.uint8), maxCorners=200, qualityLevel=0.01, minDistance=8)
    if points is None or len(points) < 10:
        return np.zeros(2)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev.astype(np.uint8), cur.astype(np.uint8), points, None)
    good = status.ravel() == 1
    if good.sum() < 10:
        return np.zeros(2)
    flow = (moved[good] - points[good]).reshape(-1, 2)
    if np.mean(np.hypot(flow[:, 0], flow[:, 1]) < 0.3) >= 0.3:
        return np.zeros(2)  # many points did not move at all, so the camera is still (only vehicles moved)
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                                  ransacReprojThreshold=1.0)
    if matrix is None or inliers.sum() < 0.5 * good.sum():  # no single motion shared by most points
        return np.zeros(2)
    return matrix[:, 2].copy()


class FrameConditioner:
    # enhance is off by default: in the benchmark (results/benchmark.json) it lowered YOLOv8 accuracy on dark images
    def __init__(self, stabilise: bool = True, enhance: bool = False) -> None:
        self.stabilise = stabilise
        self.enhance = enhance
        self._prev_small = None
        self._path = np.zeros(2)      # accumulated camera movement
        self._smooth = np.zeros(2)    # low-pass filtered camera path
        self._still_frames = 0

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 320 / gray.shape[1]
        small = cv2.resize(gray, (320, max(1, int(gray.shape[0] * scale)))).astype(np.float32)
        flags = {"low_light": False, "low_contrast": False, "blurry": False,
                 "camera_shake": False, "frozen": False, "shake_px": 0.0}

        if self._prev_small is not None and self._prev_small.shape == small.shape:
            # frozen camera: a live feed that hangs keeps sending the identical frame
            if float(np.max(np.abs(small - self._prev_small))) <= config.FROZEN_MAX_CHANGE:
                self._still_frames += 1
            else:
                self._still_frames = 0
            flags["frozen"] = self._still_frames >= config.FROZEN_FRAMES

            # camera shake: keep the camera path smooth and undo the jitter
            shift = global_shift(self._prev_small, small) / scale
            flags["shake_px"] = round(float(np.hypot(*shift)), 1)
            flags["camera_shake"] = flags["shake_px"] > config.SHAKE_PIXELS
            self._path += shift
            self._smooth = 0.9 * self._smooth + 0.1 * self._path
            correction = self._smooth - self._path
            if self.stabilise and np.hypot(*correction) > config.SHAKE_PIXELS:
                matrix = np.float32([[1, 0, correction[0]], [0, 1, correction[1]]])
                frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        self._prev_small = small

        brightness, contrast = float(gray.mean()), float(gray.std())
        flags["low_light"] = brightness < config.LOW_LIGHT_BRIGHTNESS
        flags["low_contrast"] = contrast < config.LOW_CONTRAST_STD
        flags["blurry"] = float(cv2.Laplacian(gray, cv2.CV_64F).var()) < config.BLUR_VARIANCE
        if self.enhance and (flags["low_light"] or flags["low_contrast"]):
            frame = enhance_contrast(frame, brighten=flags["low_light"])
        flags["brightness"] = round(brightness, 1)
        return frame, flags
