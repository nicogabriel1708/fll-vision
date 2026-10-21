LEFT_GPU = 0
RIGHT_GPU = 0

VIDEO_SOURCE = "match.mp4"

# ROI
TABLE_ROI = {
	"top": 0.05,  # top 5% off
	"bottom": 0.95,  # bottom 5% off
	"left": 0.05,  # left 5% off
	"right": 0.95,  # right 5% off
}

# Hand filter (geometric properties)
# Robots are cubes (Solidity > 0.8). Hands are irregular (< 0.7).
MIN_SOLIDITY = 0.75

# Minimum aspect ratio: Arms are long (Ratio < 0.4). Robots are square (0.5 - 2.0).
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5

# Minimum size in pixels
MIN_AREA = 800

# Tracking lost timeout (frames)
LOST_TIMEOUT = 30

# Resolution of the clip
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720

# How smoothly does the camera follow? (0.05 = very smooth/slow, 0.5 = direct/fast)
SMOOTH_FACTOR = 0.1
