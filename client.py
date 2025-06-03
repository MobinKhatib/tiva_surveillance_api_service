"""
Author: Mobin Khatib
Email: mbnkhatib@gmail.com
GitHub: https://github.com/MobinKhatib
Date: 2025-04-19
Description: This script is supposed to send the data to client for processing cars involving finding cars, plate and extract characters and write in Persian using API
"""
# client.py
import requests
import json # Import json module
import base64 # <-- Might need this if we're encoding a local file
import os
import json
import logging

# Load base64 image from JSON instead of .txt
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app_client.log")

# ----------------- Logging Configuration -----------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# ---------------------------------------------------------

with open(r"E:\University\Semesters\Work\Tiva_Surveillance_API\assets\aks.json", "r") as f:
    data = json.load(f)
    SAMPLE_IMAGE_BASE64 = data

# The base URL of the API server
SERVER_URL = "http://localhost:8000"

def check_server_status():
    """Sends a GET request to the server's /status endpoint."""
    status_endpoint = f"{SERVER_URL}/status"
    logging.info(f"Sending GET request to: {status_endpoint}")
    try:
        response = requests.get(status_endpoint, timeout=10) # Add a timeout

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Process the successful response
        server_data = response.json() # Parse the JSON response body
        logging.info("Received successful GET response.")
        # Pretty print the JSON data
        #print(json.dumps(server_data, indent=2))
        return server_data

    except requests.exceptions.ConnectionError:
        logging.error(f"Could not connect to the server at {SERVER_URL}.")
    except requests.exceptions.Timeout:
        logging.error("Request timed out.")
    except requests.exceptions.RequestException as e:
        # Handles HTTP errors (like 404, 500) and other request issues
        logging.error(f"GET request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Status Code: {e.response.status_code}")
            try:
                logging.error(f"Server Error Body: {e.response.text}")
            except Exception:
                pass # Ignore if body isn't text/json

# --- CHANGE: Function to send image data ---
def process_image_on_server(json_data_to_send):
    """Sends a POST request to the server's /process_plate endpoint."""
    process_endpoint = f"{SERVER_URL}/process_plate" # <-- Use the new endpoint
    logging.info(f"Sending POST request to: {process_endpoint}")


    # --- CHANGE: Payload structure ---
    payload = {
        "original_timestamp": json_data_to_send.get("original_timestamp"),
        "image_base64": json_data_to_send.get("image_base64")
    }
    # --- END CHANGE ---

    try:
        response = requests.post(process_endpoint, json=payload, timeout=100) # Increase timeout for potentially larger data/processing
        response.raise_for_status()
        # Parse JSON and return as dict
        logging.info("Received successful POST response.")
        return response.json()
    except Exception as e:
        logging.exception("Error during POST request.")
    
    return None # Indicate failure
# --- END CHANGE ---

# --- CHANGE: Main execution block ---
if __name__ == "__main__":
    logging.info("Starting API Client for Image Processing...")

    # Send the image data for processing
    processed_result = process_image_on_server(SAMPLE_IMAGE_BASE64)

    # dict
    image_base64 = None
    full_json_data = None
    
    if isinstance(processed_result, dict):
        # Already a dict, no need to parse
        full_json_data = processed_result
    plate_results = processed_result.get("plate_results") if processed_result else None
    image_base64 = processed_result.get("image_base64") if processed_result else None
    
    if full_json_data:
        os.makedirs("client_get_it_babe_folder", exist_ok=True)
        json_path = f"client_get_it_babe_folder/{plate_results}.json"
        with open(json_path, "w") as f:
            json.dump(full_json_data, f, indent=4) # Format the JSON output with an indentation of 4 spaces per level
        logging.info(f"JSON saved successfully to '{json_path}'")
    else:
        logging.warning("No valid JSON found in response.")
        
    image_base64 = processed_result.get("image_base64") if processed_result else None
    # save base64 in case of existence
    if image_base64:
        # Save the base64 string to a file
        image_path = f"client_get_it_babe_folder/{plate_results}.jpg"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        logging.info(f"Image saved successfully to '{image_path}'")
    # Save the full JSON response
# --- END CHANGE ---