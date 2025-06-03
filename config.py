# config.py - Centralized settings

import os

# Get the directory where config.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAR_MODEL_PATH = os.path.join(BASE_DIR, "model", "yolo11s.pt")
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "model", "plate.pt")
CHAR_MODEL_PATH = os.path.join(BASE_DIR, "model", "plate_character_recognizer_1.pt")

### General Settings ###
DEVICE = 'cpu'
FPS = 30  # Default FPS

### Processing Regions ###
CAR_REGION = [(0, 100), (1280, 100)]  # Line for speed measurement
PLATE_REGION = [(100, 200), (1250, 720)]  # Rectangle for plate detection
REGION = [(0, 200), (1280, 720)]
### Feature Toggles ###
ENABLE_PLATE_CROP = True
ENABLE_PLATE = True  # Set to False to disable plate detection
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence to save detected plates
ENABLE_WATERMARK = False  # Set False to disable watermark
WATERMARK_COLOR = "black"  # Options: "white", "black"
# Real Address
ENABLE_ADDRESS = True
Address = ""
# Geographic Coordinates
ENABLE_GEOGRAPHIC_COORDINATES = True
Geographic_Coordinates = ""
#Date
ENABLE_DATE = True
DATE = ""
# Location
ENABLE_LOCATION = True
LOCATION =  ""
# date on the text
#ENABLE_DATE_TEXT = False

# Logger 
LEVEL = "DEBUG"  # or "INFO", "WARNING", etc.

# IP and Port to Server
MY_IP = '192.168.1.6'
HOST = '0.0.0.0' # or it can be localhost or anything else
PORT = 5000
