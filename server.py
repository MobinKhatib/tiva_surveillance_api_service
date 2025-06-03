"""
Author: Mobin Khatib
Email: mbnkhatib@gmail.com
GitHub: https://github.com/MobinKhatib
Date: 2025-04-19
Description: This script is processing cars involving finding cars, plate and extract characters and write in Persian using API
"""
# server.py
from flask import Flask, request, jsonify
import time
import numpy as np
import plate # Assuming your PlateDetector class is in plate.py
import cv2
import base64
import config
from collections import OrderedDict
import json
from flask import Response
# Redirect Flask's default loggers to use your config
import logging
flask_log = logging.getLogger('flask.app')
werkzeug_log = logging.getLogger('werkzeug')

flask_log.setLevel(logging.INFO)
werkzeug_log.setLevel(logging.INFO)

logging.info("Initializing Plate Detector...")
try:
    plate_detector = plate.PlateDetector(
        car_model_path=config.CAR_MODEL_PATH,
        plate_model_path=config.PLATE_MODEL_PATH,
        char_model_path=config.CHAR_MODEL_PATH,
        device='cpu', # Change to 'cuda' if you have a GPU and compatible PyTorch/CV2
        region=[(0, 0), (1280, 720)] # Example region
    )
    logging.info("Plate Detector Initialized Successfully.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not initialize Plate Detector: {e}")
    # Depending on your setup, you might want to exit or handle this differently
    exit() # Or raise SystemExit

# Create a Flask application instance
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  #  Add this line to preserve dict order

# In-memory storage (replace with database in real applications)
received_messages = []


# --- API Endpoints ---
# Endpoint to GET some data (e.g., server status or a simple message)
@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current status of the server."""
    logging.info("GET request received for /status")
    response_data = {
        "status": "OK",
        "message": "Server is running smoothly!",
        "timestamp": time.time(),
        "received_message_count": len(received_messages)
    }
    return jsonify(response_data), 200 # 200 OK status code

# Endpoint to POST (send) data to the server
@app.route('/process_plate', methods=['POST']) # <-- Renamed endpoint
def handle_plate_image(): # <-- Renamed function
    """Receives timestamp and base64 image, processes it, and returns processed base64."""
    logging.info("POST request received for /process_plate")

    data = request.get_json()

    if not data:
        logging.warning("No JSON data received")
        return jsonify({"error": "Request must be JSON"}), 400

    # --- CHANGE: Expect 'timestamp' and 'image_base64' ---
    received_timestamp = data.get('original_timestamp')
    received_base64 = data.get('image_base64')

    if not received_timestamp or not received_base64:
        logging.warning("Missing 'timestamp' or 'image_base64' in request")
        return jsonify({"error": "'timestamp' and 'image_base64' fields are required"}), 400
    # --- END CHANGE ---

    logging.info(f"Received image data (timestamp: {received_timestamp})")
    # print(f"Base64 data (truncated): {received_base64[:50]}...") # Optional: print start of base64

    try:
        # --- CHANGE: Call your plate.py processing function ---
                # --- Decode Base64 to OpenCV Image ---
        img_bytes = base64.b64decode(received_base64)
        img_np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("Could not decode base64 image data")
            return jsonify({"error": "Invalid image data received"}), 400

        logging.info(f"Decoded image shape: {frame.shape}")    
        #config.DATE = received_timestamp # this is the default timestamp, it'll set by clients
        plate_data, encoded_image = None, None

        result = plate_detector.detect_plates(frame)
        recognized_plates = result.get("recognized_plates", [])
        encoded_image = result.get("base64_image", None)

        if recognized_plates:
            plate_data = recognized_plates[0][0]
        else:
            plate_data = None


        # --- CHANGE: Prepare response with processed data ---
        response_data = OrderedDict([
            ("status", "success"),
            ("original_timestamp", received_timestamp),
            ("plate_results", plate_data),
            ("image_base64", encoded_image)
        ])
        
        # return jsonify(response_data), 200 # 200 OK is suitable here # this was change the order so I do below instead
        
        response_json = json.dumps(response_data)  # preserves OrderedDict order
        return Response(response=response_json, status=200, mimetype='application/json')

        # return jsonify(response_data), 200 # 200 OK is suitable here
        # --- END CHANGE ---

    except Exception as e:
        # --- ADD: Error handling for processing ---
        logging.exception(f"Error during image processing: {e}")
        return jsonify({"error": "Processing failed on server", "details": str(e)}), 500 # 500 Internal Server Error
        # --- END ADD ---

# --- Run the Server ---
if __name__ == '__main__':
    logging.info("Starting Flask server...")
    # host='0.0.0.0' makes the server accessible from other machines on the network
    # Use 'localhost' or '127.0.0.1' to only allow connections from the same machine
    # debug=True enables auto-reloading and detailed error pages (DON'T use in production)
    #app.run(host='localhost', port=8000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

    #app.run(host='192.168.1.109', port=5000, debug=True)
