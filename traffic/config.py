"""
All tunable numbers in one place.

The values are reasonable defaults for a single road approach seen by one
camera. They depend on the camera angle and road size, so adjust them for a
real junction (the dashboard lets you change the most important ones).
"""

# ---- Detection -------------------------------------------------------------
# COCO category names counted as vehicles (YOLOv8 and Faster R-CNN both use COCO)
VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

# Class-name keywords that mean "emergency vehicle" in a custom-trained model
EMERGENCY_KEYWORDS = ("ambulance", "fire", "police", "emergency")

DEFAULT_CONFIDENCE = 0.4

# ---- Lanes and density -----------------------------------------------------
DEFAULT_LANES = 3  # the frame is split into this many equal vertical lanes

# Vehicles per lane: Low < 5, Medium 5-9, High 10-14, Very High >= 15
DENSITY_THRESHOLDS = (5, 10, 15)
DENSITY_LEVELS = ("Low", "Medium", "High", "Very High")

SMOOTHING_WINDOW = 15  # frames; density uses the median count of the last N frames

# ---- Road clearance time ---------------------------------------------------
# Passenger-car-unit style weights: a bus takes longer to clear than a motorcycle.
VEHICLE_WEIGHTS = {"car": 1.0, "motorcycle": 0.5, "bus": 2.0, "truck": 2.0, "emergency": 1.5}
SPACING_M = 7.0           # road length one car-equivalent occupies in a queue (car + gap)
STARTUP_LOST_TIME_S = 2.0  # time lost while the first vehicles start moving
MIN_SPEED_MPS = 2.0        # floor so a stopped queue does not give an infinite time

# Speed used when it cannot be measured (no calibration or too few tracks), m/s
DEFAULT_SPEED_BY_DENSITY = {"Low": 8.0, "Medium": 6.0, "High": 4.0, "Very High": 3.0}

# Metres per pixel along the road. 0 means "not calibrated": speed is then
# taken from DEFAULT_SPEED_BY_DENSITY instead of being measured.
METERS_PER_PIXEL = 0.0

# ---- Signal timing ---------------------------------------------------------
YELLOW_S = 5
MIN_GREEN_BY_DENSITY = {"Low": 15, "Medium": 25, "High": 40, "Very High": 55}
MAX_GREEN_S = 90
CYCLE_S = 120             # red = cycle - green - yellow (other approaches get the rest)
MIN_RED_S = 10
EMERGENCY_GREEN_S = 90
EMERGENCY_RED_S = 10

# ---- Alerts ----------------------------------------------------------------
ALERT_COOLDOWN_S = 30  # at most one alert per 30 s for the same emergency

# ---- Edge cases (image quality) ---------------------------------------------
LOW_LIGHT_BRIGHTNESS = 60   # mean brightness (0-255) below this = low light
LOW_CONTRAST_STD = 35       # grey-level std below this = fog, rain or haze
BLUR_VARIANCE = 50          # Laplacian variance below this = blurry frame
SHAKE_PIXELS = 2.0          # frame shift above this = camera shake (corrected)
FROZEN_MAX_CHANGE = 1.0     # no pixel changed by more than this = the same frame was repeated
FROZEN_FRAMES = 30          # this many unchanged frames in a row = camera frozen

# ---- Tracking (occlusion handling) -------------------------------------------
TRACK_IOU_MATCH = 0.3
TRACK_MAX_MISSES = 15   # keep a hidden vehicle for 15 frames before dropping it
TRACK_MIN_HITS = 3      # a track must be seen 3 times before it is counted

# ---- Storage ---------------------------------------------------------------
DB_PATH = "data/traffic.db"
LOG_INTERVAL_S = 1.0  # write one row per lane every second of video
