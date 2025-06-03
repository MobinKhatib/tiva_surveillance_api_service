"""
Author: Mobin Khatib
Email: mbnkhatib@gmail.com
GitHub: https://github.com/MobinKhatib
Date: 2025-04-19
Description: This script is the main code for processing cars involving finding cars, plate and extract characters and write in Persian ( the server use this code to process)
"""
import cv2
from ultralytics import YOLO
import os
from persiantools.jdatetime import JalaliDate
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from logger import logger
import config
import base64
from contextlib import contextmanager
from logger import logger, log_exceptions, log_context, log_exceptions_all_methods
import arabic_reshaper
from bidi.algorithm import get_display


logger.info(" plate.py logger is working!")

@log_exceptions_all_methods
class PlateDetector:
    def __init__(self, car_model_path=config.CAR_MODEL_PATH, plate_model_path=config.PLATE_MODEL_PATH
                 , char_model_path=config.CHAR_MODEL_PATH
                 , device='cpu'
                 , region=config.REGION
    ):
        """Initialize the license plate detection model."""
        self.device = device  # Store the device choice

        with log_context("Car model loading"):
            self.car_model_path = YOLO(car_model_path).to(self.device)
        with log_context("Plate model loading"):
            self.plate_model_path = YOLO(plate_model_path).to(self.device)
        with log_context("Char model loading"):
            self.char_model_path = YOLO(char_model_path).to(self.device)
            
        self.best_frames = {}  # Store best frames across multiple calls

        # Store the region settings
        self.region = region
        logger.info(f"Detection region set to: {self.region}")
                
    #@staticmethod  # Since format_plate_number() does not use self, mark it as @staticmethod so it doesn't receive self
    def format_plate_number(self, chars):

        #logger.info("Formatting plate number with received characters: %s", chars)
        #logger.debug("Raw input characters (x1, char, conf): %s", chars)

        if not chars:  # If chars list is empty
            logger.warning("No characters detected for plate formatting.")
            return "@@@@@@@@", 0.0  # Return default plate + confidence 0.0

        # Define the final plate structure with placeholders '@'
        formatted_plate = ["@"] * 8  
        formatted_confidence = [0.0] * 8  # Default confidence scores

        # Sort characters by x1 (left-to-right order)
        # Step 1: Sort by confidence score (high to low)
        chars.sort(key=lambda x: x[2], reverse=True)  # Sorting by confidence (index 2)
        logger.debug("Characters sorted by confidence: %s")

        # Step 2: If there are more than 8 characters, keep only the top 8 by confidence
        if len(chars) > 8:
            chars = chars[:8]
            logger.info("More than 8 characters detected, keeping only top 8 by confidence.")
            
            #logger.debug("Characters sorted by x1 position: %s")

        # Step 3: After filtering, sort by x1 position to maintain left-to-right order
        chars.sort(key=lambda x: x[0])  # Sort by x1 (index 0) to restore left-to-right order

        logger.debug("Characters sorted by x1 position: %s")

        numbers = [(c, conf) for x1, c, conf in chars if isinstance(c, str) and c.isdigit()]
        letters = [(c, conf) for x1, c, conf in chars if isinstance(c, str) and not c.isdigit()]
        # If 'SAD' (or any other non-digit string) is part of the plate
        # Apply proper handling based on your formatting needs
        # Choose the most confident letter for the 3rd slot
        if letters:
            best_letter, best_conf = max(letters, key=lambda x: x[1])  # Choose letter with highest confidence
            logger.debug("Selected letter for 3rd slot: %s (confidence: %.2f)", best_letter, best_conf)
            formatted_plate[2] = best_letter  
            formatted_confidence[2] = best_conf  # Store confidence
        else:
            formatted_plate[2] = '@'  # Default placeholder if no letter is detected
            logger.warning("No valid letter detected for the 3rd position(letter) in plate.")

        # Expected digit positions
        digit_positions = [0, 1, 3, 4, 5, 6, 7]  

        # Step 1: Keep numbers in original order but remove lowest confidence ones if more than 7
        if len(numbers) > 7:
            logger.info("More than 7 digits detected, filtering based on confidence.")
            logger.debug("Digits after filtering by confidence: %s", numbers)
            numbers.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence (high to low)
            numbers = sorted(numbers[:7], key=lambda x: x[1])  # Restore original order of remaining numbers
        
        # Put numbers with plates
        for pos, (num, conf) in zip(digit_positions, numbers):
            formatted_plate[pos] = num
            formatted_confidence[pos] = conf  # Store confidence
        # Debugging: Check final plate format before returning
        i, average_confidence = 0, 0
        for i in range(len(formatted_confidence)):
            average_confidence = formatted_confidence[i] + average_confidence
        average_confidence = average_confidence / 8
        if any(c == "@" for c in formatted_plate):
            logger.warning("Some plate positions could not be filled properly.")

        logger.info("Final formatted plate: %s with confidence: %.2f", "".join(formatted_plate), average_confidence)
        logger.debug("Formatted plate string: %s", "".join(formatted_plate))
        logger.debug("Average confidence: %.2f", average_confidence)
        
        return "".join(formatted_plate), average_confidence  # Convert list to string
    
    def recognize_plate_chars(self, frame, plates):
        """Runs character recognition on detected plates."""
        recognized_plates = []

        for (x1, y1, x2, y2, id) in plates:
            logger.info(" Processing plate ID: %s | Coordinates: (%d, %d, %d, %d)", id, x1, y1, x2, y2)
            try:
                plate_crop = frame[y1:y2, x1:x2]  # Crop plate region
                plate_crop = cv2.resize(plate_crop, (160, 64)) # 160,64 # Resize for OCR model
                logger.debug("Plate cropped and resized for OCR.")
                #plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
                results = self.char_model_path(plate_crop)  # Run character recognition
                chars = []
                charclassnames = [    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
'ALEF', 'BE', 'PE', 'TE', 'SE', 'JIM', 'DAL', 'ZHE',
'SIN', 'SAD', 'TA', 'EIN', 'GHAF', 'LAM', 'MIM', 'NON',
'VAV', 'HEH', 'YE', 'D', 'S']
                for result in results:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        x1, y1, x2, y2 = map(int, box)
                        char = charclassnames[int(cls)]  # Correct way to map class index to character
                        chars.append((x1, char, float(conf)))  # Store confidence score
                    logger.debug("Extracted characters (x1, char, conf): %s", chars)

            
                # Sort characters by x-coordinates to maintain order
                chars.sort()
                #plate_number = ''.join([char for _, char in chars])
                plate_number = ''.join([char for x1, char, conf in chars])  # Extract only characters
               
                logger.debug("Sending characters to formatter.")
                #plate_number = self.format_plate_number([char for _, char in chars])  # Apply formatting
                plate_number, average_confidence = self.format_plate_number([(x1, char, conf) for x1, char, conf in chars])  # Send both char and conf
            
                #recognized_plates.append((plate_number, id))
                recognized_plates.append((plate_number, id, average_confidence))            
                logger.info(" Plate recognized: %s | Confidence: %.2f", plate_number, average_confidence)

            except Exception as e:
                logger.error(f"Error occurred while processing plate for ID {id}: {e}")
                logger.debug("Continuing to next plate after failure on ID %s", id)
                continue  # Continue processing next plates if one fails

        return recognized_plates
    def find_vehicle_in_frame_and_get_id_vehicle(self, frame):
        """
        Returns a dictionary of detected vehicle objects.
        Key: sequential persistent ID
        Value: dict with 'bbox', 'class', and 'confidence'
        """
        vehicle_dict = {}

        # Run detection
        results = self.car_model_path(frame)
        if not results[0].boxes:
            logger.warning("No cars detected in the frame.")
        else:
            logger.info("Car detected with coordinates: %s", results[0].boxes.xyxy)

        result = results[0]  # assuming one image # we only path one image by default, this is unncessary!
        target_classes = {2, 3, 5, 7} # If It's my pre-trained model! zero is car yolo model: 2 car 5 bus 7 truck
        boxes = result.boxes.xyxy.cpu()
        classes = result.boxes.cls.cpu()
        confidences = result.boxes.conf.cpu()
        
        max_conf = 0
        best_detection = None
        
        for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
            if int(cls_id) in target_classes and conf > max_conf:
                    max_conf = conf

                    best_detection = {
                    'bbox': box.tolist(),
                    'class': int(cls_id),
                    'confidence': float(conf)
                }
        if best_detection:
            vehicle_dict[0] = best_detection
                
        return vehicle_dict

    def detect_plates(self, frame):
    
        plates = []    
        logger.info(" Starting plate detection on frame.")
        #  Get vehicle data from 
        vehicle_dict = self.find_vehicle_in_frame_and_get_id_vehicle(frame)

        allowed_labels = {1}  # just allowed labels (plate, platefreezone)

        for result in self.plate_model_path(frame): # this is the model for plates
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  # get classes with boxes
                if int(cls) not in allowed_labels: # check if it is allowed
                    continue  # if not allowed, ignore

                x1, y1, x2, y2 = map(int, box)
                logger.debug("Detected plate box: (%d, %d, %d, %d)", x1, y1, x2, y2)
                plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Find the closest id vehicle
                closest_vehicle_id = None
                min_distance = float("inf")

                #for car_id, (vx1, vy1, vx2, vy2) in vehicle_dict.items():
                for car_id, vehicle_info in vehicle_dict.items():
                    vx1, vy1, vx2, vy2 = map(int, vehicle_info['bbox'])
                    car_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
                    distance = ((plate_center[0] - car_center[0]) ** 2 + (plate_center[1] - car_center[1]) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        logger.debug("Plate matched to vehicle ID: %s with distance: %.2f", car_id, distance)
                        closest_vehicle_id = car_id
            
                # Store the plate with its matched vehicle ID
                plates.append((x1, y1, x2, y2, closest_vehicle_id))
                logger.info("Plate associated with vehicle ID %s added to processing list.", closest_vehicle_id)
        
        # Recognize characters in plates
        recognized_plates = self.recognize_plate_chars(frame, plates)      
        logger.info("Character recognition completed on detected plates.")
        
        # Combine coordinates
        plates_coordinates = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in plates]
        combined_data = [
            (plate_number, id, average_confidence, coords)
            for (plate_number, id, average_confidence), coords in zip(recognized_plates, plates_coordinates)
        ]

        # Filter by region and pick best
        best_data = None
        y1_threshold, y2_threshold = self.region[0][1], self.region[1][1]
        x1_threshold, x2_threshold = self.region[0][0], self.region[1][0]
        
        for plate_number, id, average_confidence, plates_coordinates in combined_data:


            if y1 > y1_threshold and y2 < y2_threshold and x1 > x1_threshold and x2 < x2_threshold:
                #  Keep only the highest confidence frame for each id
                car_coordinates = vehicle_dict.get(id, (None, None, None, None))  #  Define before use
                #print("lalalala\n", car_coordinates)
                best_data = {
                    'frame': frame.copy(),
                    'confidence': average_confidence,
                    'plate_number': plate_number,
                    'plates_coordinates': (x1, y1, x2, y2),
                    'car_coordinates': car_coordinates
                }
            break  # pick first valid — you can adjust logic here
        if best_data:
            base64_img = self.draw_vehicle_box_and_watermark_frames(frame, best_data, vehicle_dict)
        else:
            base64_img = None

        return {
            'recognized_plates': recognized_plates,
            'base64_image': base64_img
        }
    def reshape_persian_text(self, text):
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text

    def draw_vehicle_box_and_watermark_frames(self, frame, best_data, vehicle_dict): 
        #vehicle_dict = self.find_vehicle_in_frame_and_get_id_vehicle(frame)  # Get id vehicle data
        logger.info(" Starting processing for best frames.")

        plate_number = best_data['plate_number']
        persian_plate = self.convert_to_persian(plate_number)
        confidence = int(best_data['confidence'] * 100 + 1)

        if confidence < config.CONFIDENCE_THRESHOLD:
            logger.info("Skipped plate %s due to low confidence (%d%%)", plate_number, confidence)
            return None
        
        #vehicle_dict = self.find_vehicle_in_frame_and_get_id_vehicle(frame)
        car_coordinates = self._get_valid_car_coordinates(vehicle_dict, best_data)
        #print("car coordinates la la la la\n" , car_coordinates)
        if car_coordinates:
            self._draw_vehicle_box(frame, vehicle_dict)

        plate_crop = self._prepare_plate_crop(frame, best_data)
        if plate_crop is not None and config.ENABLE_WATERMARK:
            new_frame = self._create_frame_with_overlay(frame, plate_crop)
            self._draw_text_on_frame(new_frame, persian_plate)
            logger.info("Watermarked frame generated for plate %s", plate_number)

        else:
            new_frame = frame

        return self.encode_frame_to_base64(new_frame)
    
    def _get_valid_car_coordinates(self, vehicle_dict, best_data):
        stored_coordinates = best_data.get("car_coordinates", (None, None, None, None))
        id = best_data.get("id")  # optional: if you're tagging ID for logging

        car_coordinates = stored_coordinates
        if car_coordinates == (None, None, None, None):
            logger.warning("No valid stored car coordinates.")
            # Try to fallback if needed
            car_coordinates = vehicle_dict.get(id, (None, None, None, None))

        return car_coordinates

    def _draw_vehicle_box(self, frame, vehicle_dict):
        for car_id, car_data in vehicle_dict.items():
            try:
                vx1, vy1, vx2, vy2 = map(int, car_data['bbox'])
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            except Exception as e:
                logger.warning("Skipping frame due to invalid bbox for car_id=%s: %s", car_id, e)

    def _prepare_plate_crop(self, frame, best_data):
        x1, y1, x2, y2 = best_data.get("plates_coordinates", (0, 0, 0, 0))
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop is not None and plate_crop.size > 0:
            return cv2.resize(plate_crop, (200, 42))
        else:
            logger.warning("Plate crop is empty or invalid.")
            return None

    def _create_frame_with_overlay(self, frame, plate_crop):
        logger.info("Creating overlay frame for watermarking.")
        if config.ENABLE_WATERMARK:
            new_height = 80
            border_color = (0, 0, 0) if config.WATERMARK_COLOR == "black" else (255, 255, 255)
            new_frame = cv2.copyMakeBorder(frame, new_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=border_color)
        else:
            new_frame = frame.copy()
        logger.info("Watermark border added with height %d.", new_height)

        if config.ENABLE_PLATE_CROP:
            h, w = plate_crop.shape[:2]
            new_frame[15:15+h, 10:10+w] = plate_crop
            logger.debug("Plate crop added to new frame at position (15, 10). Size: (%d, %d)", h, w)

            # ✅ NEW: Load the image you want to overlay (e.g., logo.jpg)
        overlay_path = os.path.join("Fonts", "emptyplate.jpg")  # update folder/image name
        overlay_img = cv2.imread(overlay_path)

        if overlay_img is not None:
            # Resize to fit 100x40 (450-350 width and 60-20 height)
            overlay_resized = cv2.resize(overlay_img, (230, 42))  # (width, height)
            logger.debug("Overlay image '%s' loaded and resized to (230, 42).", overlay_path)

            # Place overlay at (60, 350)
            new_frame[15:57, 230:460] = overlay_resized
        else:
            logger.warning("Overlay image not found at path: %s", overlay_path)
        logger.info("Overlay frame created successfully.")
        return new_frame
    
    def _draw_text_on_frame(self, frame, persian_plate):
        logger.info("Drawing text and metadata on frame.")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font_path = "Fonts/YEKAN.ttf"
        font = ImageFont.truetype(font_path, 20)
        font_plate = ImageFont.truetype(font_path, 30)
        logger.debug("Loaded font from: %s", font_path)

        img_width = self.region[1][0] - self.region[0][0]
        x_offset = self.region[0][0]

        fill_text = (255, 255, 255) if config.WATERMARK_COLOR == "black" else (0, 0, 0)

        if config.ENABLE_DATE:
            draw.text((img_width - 20, x_offset + 20), self.reshape_persian_text("تاریخ:"), font=font, fill=fill_text, anchor="ra")
            draw.text((img_width - 100, x_offset + 40), f"{config.DATE}", font=font, fill=fill_text, anchor="ra")

            logger.debug("Date label drawn.")
        if config.ENABLE_LOCATION:
            draw.text((img_width - 20, x_offset + 40), self.reshape_persian_text("نام محل:"), font=font, fill=fill_text, anchor="ra")
            draw.text((img_width - 100, x_offset + 40), f"{config.LOCATION}", font=font, fill=fill_text, anchor="ra")
            logger.debug("Location label drawn with value: %s", config.LOCATION)
        if config.ENABLE_ADDRESS:
            draw.text((img_width // 2 + 200, x_offset + 20), self.reshape_persian_text("آدرس:"), font=font, fill=fill_text, anchor="ra")
            draw.text((img_width // 2 + 100, x_offset + 20), f"{config.Address}", font=font, fill=fill_text, anchor="ra")
            logger.debug("Address label drawn with value: %s", config.Address)
        if config.ENABLE_GEOGRAPHIC_COORDINATES:
            draw.text((img_width // 2 + 200, x_offset + 40), self.reshape_persian_text("مختصات:"), font=font, fill=fill_text, anchor="ra")
            draw.text((img_width // 2 + 100, x_offset + 40), f"{config.Geographic_Coordinates}", font=font, fill=fill_text, anchor="ra")
            logger.debug("Geographic coordinates label drawn with value: %s", config.Geographic_Coordinates)

        if config.ENABLE_PLATE:
            char1, char2, char3, char4, char5, char6, char7, char8 = persian_plate
            plate_text = self.reshape_persian_text(f"{char7}{char8}   {char4}{char5}{char6} {char3} {char1}{char2}")
            draw.text((450, 22), plate_text, font=font_plate, fill=(0, 0, 0), anchor="ra")
            logger.debug("Formatted Persian plate text drawn: %s", plate_text)

        new_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:new_frame.shape[0], :new_frame.shape[1]] = new_frame
        logger.info("Text drawing on frame completed.")
    
    def convert_to_persian(self, plate_number):
        replacements = {
            "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "0": "0",
            "ALEF": "الف", "BE": "ب", "PE": "پ", "TE": "ت", "SE": "ث", "JIM": "ج", "DAL": "د", "ZHE": "ژ",
            "SIN": "س", "SAD": "ص", "TA": "ط", "EIN": "ع", "GHAF": "ق", "LAM": "ل", "MIM": "م", "NON": "ن",
            "VAV": "و", "HEH": "ه", "YE": "ی", "D": "D", "S": "S"
            # Add more replacements as needed
        }
        for eng, fa in replacements.items():
            plate_number = plate_number.replace(eng, fa)
        #logger.debug("Converted Persian plate: %s", plate_number)    
        return plate_number
                        
    def encode_frame_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        logger.debug("Encode frame to base64.")
        return base64.b64encode(buffer).decode('utf-8')            