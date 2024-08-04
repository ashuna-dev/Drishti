import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import cv2
import math
import google.generativeai as genai
from imutils.video import WebcamVideoStream, FPS
from pathlib import Path
import os
from gtts import gTTS
from deep_translator import GoogleTranslator


class TrackedObject:
    def __init__(self, cls, coords):
        self.cls = cls
        self.coords = coords

class BlindAssistant:
    def __init__(self):
        # Initialize video stream and YOLO model
        #self.src = src
        self.cap = WebcamVideoStream(src=0).start()
        self.model = YOLO("yolov8n.pt")
        self.fps = FPS().start()
        self.analysis_text = ""

        
        

        # Define object classes and other parameters
        self.classNames =["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
        
        self.object_real_height = {"person": 1.75, "bicycle": 1.0, "car": 1.5, "motorbike": 1.0, "aeroplane": 2.0, 
                                 "bus": 3.0, "train": 4.0, "truck": 2.5, "boat": 2.0, "traffic light": 0.5, 
                                 "fire hydrant": 0.5, "stop sign": 0.5, "parking meter": 0.5, "bench": 1.0, 
                                 "bird": 0.2, "cat": 0.3, "dog": 0.3, "horse": 1.6, "sheep": 1.0, "cow": 1.2, 
                                 "elephant": 2.5, "bear": 1.5, "zebra": 1.5, "giraffe": 3.0, "backpack": 0.4, 
                                 "umbrella": 1.0, "handbag": 0.4, "tie": 0.2, "suitcase": 0.6, "frisbee": 0.2, 
                                 "skis": 1.5, "snowboard": 1.5, "sports ball": 0.3, "kite": 0.6, "baseball bat": 0.8,
                                 "baseball glove": 0.2, "skateboard": 0.8, "surfboard": 1.5, "tennis racket": 0.8,
                                 "bottle": 0.2, "wine glass": 0.2, "cup": 0.6, "fork": 0.2, "knife": 0.2, 
                                 "spoon": 0.2, "bowl": 0.8, "banana": 0.2, "apple": 0.2, "sandwich": 0.8, 
                                 "orange": 0.4, "broccoli": 0.4, "carrot": 0.4, "hot dog": 0.4, "pizza": 0.6, 
                                 "donut": 0.2, "cake": 0.6, "chair": 1.0, "sofa": 2.0, "potted plant": 0.4, 
                                 "bed": 2.0, "dining table": 1.0, "toilet": 0.8, "tv monitor": 1.5, "laptop": 0.3, 
                                 "mouse": 0.1, "remote": 0.1, "keyboard": 0.2, "cell phone": 0.2, "microwave": 0.6, 
                                 "oven": 0.6, "toaster": 0.4, "sink": 0.4, "refrigerator": 1.5, "book": 0.2, 
                                 "clock": 0.3, "vase": 0.4, "scissors": 0.2, "teddy bear": 0.4, "hair drier": 0.2, 
                                 "toothbrush": 0.2}
        
        # Define intervals
        self.speech_interval = 6
        self.analysis_interval = 10

        # Initialize last speech and analysis times
        self.last_speech_time = time.time()
        self.last_analysis_time = time.time()

        
        '''self.camera_height = 1.5  # meters
        self.pitch_angle = math.radians(45)  # Convert pitch angle to radians
        self.vfov = math.radians(60)  # Vertical field of view in radians
        self.image_height = 480  # Image height
        self.image_width = 640  # Image width'''
        self.focal_length = 558
        self.tracked_objects = {}

        
        # Configure Google Generative AI
        self.api_key = 'AIzaSyAHR1jdjWT1CF3rGoNhyRmJbEWjbi5GMMw'  # Replace with your API key
        genai.configure(api_key=self.api_key)
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048
        }
        self.model_genai = genai.GenerativeModel("gemini-1.5-pro", generation_config=generation_config)

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        
        
        self.moving_avg_window = 5  # Adjust window size as needed
        self.moving_avg_distances = {cls: [] for cls in self.classNames}
        
    
    
    def calculate_moving_avg(self, cls, distance):
        # Append the new distance to the moving average list
        self.moving_avg_distances[cls].append(distance)

        # If the list exceeds the window size, remove the oldest element
        if len(self.moving_avg_distances[cls]) > self.moving_avg_window:
            self.moving_avg_distances[cls].pop(0)

        # Calculate the moving average
        moving_avg = sum(self.moving_avg_distances[cls]) / len(self.moving_avg_distances[cls])

        return moving_avg

    def calculate_direction(self, object_center_x):
        frame_width = self.cap.frame.shape[1]

        # Calculate the direction of the object from the center of the frame
        if object_center_x < frame_width // 3:
            direction = "right"
        elif object_center_x > 2 * frame_width // 3:
            direction = "left"
        else:
            direction = "center"
        
        return direction
    def speak_text(self, text,lang='en'):
    # Initialize gTTS with the text
        translated_text = GoogleTranslator(source='auto', target=lang).translate(text)

        # Initialize gTTS with the translated text
        tts = gTTS(text=translated_text, lang=lang)
        
    # Save the speech to a temporary file
        tts_path = "speech.mp3"
        tts.save(tts_path)

    # Use a media player to play the speech
        subprocess.run(["wmic", "process", "call", "create", '"wmplayer.exe /play /close ' + tts_path + '"'])


    def analyze_image(self, img):
        # Save the image temporarily for analysis
        
        image_path = "image.jpg"
        cv2.imwrite(image_path, img)

        # Load the image for analysis
        image_part = {
            "mime_type": "image/jpeg",
            "data": Path(image_path).read_bytes()
        }

        # Prompt the AI to describe the image
        prompt_parts = ["describe image and ignore boxes and text over it, also any fps. make it read in natural language so that a blind person is able to understand the enviroment. If a person is present describe its emotions.", image_part]
        try:
            response = self.model_genai.generate_content(prompt_parts)

            # Check the response for correctness
            if response and response.text:
                
                
                # Print the response text
                print("Analysis response:", response.text)
                self.analysis_text = response.text


                # Speak the response text    
                self.speak_text(response.text)
            else:
                print("No valid response received from generative AI.")
        except Exception as e:
            print(f"Error during analysis: {e}")

        # Clean up the temporary image file
        os.remove(image_path)

    def apply_nms(self, detections, iou_threshold):
        # Sort detections by confidence score (descending order)
        detections.sort(key=lambda x: x[1], reverse=True)
        # Initialize list to store filtered detections
        filtered_detections = []
        while detections:
            # Select detection with highest confidence score
            max_conf_detection = detections[0]
            filtered_detections.append(max_conf_detection)
            # Calculate IoU with other detections
            detections = [d for d in detections if self.calculate_iou(max_conf_detection, d) <= iou_threshold]
        return filtered_detections

    def calculate_iou(self, box1, box2):
        # Calculate intersection coordinates
        intersection_x1 = max(box1[3][0], box2[3][0])
        intersection_y1 = max(box1[3][1], box2[3][1])
        intersection_x2 = min(box1[3][2], box2[3][2])
        intersection_y2 = min(box1[3][3], box2[3][3])
        # Calculate intersection area
        intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
        # Calculate union area
        box1_area = (box1[3][2] - box1[3][0] + 1) * (box1[3][3] - box1[3][1] + 1)
        box2_area = (box2[3][2] - box2[3][0] + 1) * (box2[3][3] - box2[3][1] + 1)
        union_area = box1_area + box2_area - intersection_area
        # Calculate IoU
        iou = intersection_area / union_area
        return iou

    def refine_bounding_boxes(self, detections):
        refined_detections = []
        for detection in detections:
            refined_detection = detection  # Placeholder for refinement (adjust as needed)
            refined_detections.append(refined_detection)
        return refined_detections

    def generate_frames_with_audio(self):
        while True:
            img = self.cap.read()
            if img is None or img.size == 0:
                continue  # Skip invalid frames

            results = self.model(img, stream=True)

            objects = []
            # Process detections and sort them by distance (closest to furthest)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    box_width = x2 - x1
                    box_height = y2 - y1
                    object_size_in_frame = max(box_width, box_height)
                    real_size = self.object_real_height[self.classNames[cls]]
                    distance = (real_size * self.focal_length) / object_size_in_frame
                    distance = round(distance, 2)
                    smoothed_distance = self.calculate_moving_avg(self.classNames[cls], distance)

                    objects.append((cls, confidence, smoothed_distance, (x1, y1, x2, y2)))
             
             
                    
            objects = self.apply_nms(objects, iou_threshold=0.5)

            # Refine bounding boxes
            objects = self.refine_bounding_boxes(objects)
            
            # Sort objects by distance (closest first)
            objects.sort(key=lambda obj: obj[2])

            # Sort objects by distance (closest first)

            # Prepare text to speak
            text_to_speak = []
            for i, obj in enumerate(objects, 1):
                cls, confidence, distance, coords = obj
                x1, y1, x2, y2 = coords
                direction = self.calculate_direction((x1 + x2) // 2)

                # Create shorter phrases
                label = f"{self.classNames[cls]} at {distance:.2f} meters, {direction}"
                text_to_speak.append(label)

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Update or create tracked object
                if cls in self.tracked_objects:
                    self.tracked_objects[cls].coords = coords
                else:
                    self.tracked_objects[cls] = TrackedObject(cls, coords)

            # Get the current time
            current_time = time.time()

            # Speak the accumulated text if the speech interval has passed
            if current_time - self.last_speech_time >= self.speech_interval:
                if text_to_speak:
                    # Submit the TTS task to the thread pool executor for asynchronous execution
                    self.executor.submit(self.speak_text, "\n".join(text_to_speak))
                self.last_speech_time = current_time

            # Analyze the image if the analysis interval has passed
            if current_time - self.last_analysis_time >= self.analysis_interval:
                # Submit the image analysis task to the thread pool executor for asynchronous execution
                self.executor.submit(self.analyze_image, img)
                self.last_analysis_time = current_time

            # Get the current time after processing
            current_time_after_processing = time.time()

            # Calculate time taken for processing
            processing_time = current_time_after_processing - current_time

            # Delay the loop to match the desired FPS
            #delay_time = max(1 / self.desired_fps - processing_time, 0)
            #time.sleep(delay_time)

            # Yield the processed frame
            _, buffer = cv2.imencode('.jpg', img)
            img_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        # Main loop
        while True:
            img = self.cap.read()
            if img is None or img.size == 0:
                continue  # Skip invalid frames

            results = self.model(img, stream=True)

            objects = []
            # Process detections and sort them by distance (closest to furthest)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    box_width = x2 - x1
                    box_height = y2 - y1
                    object_size_in_frame = max(box_width, box_height)
                    real_size = self.object_real_height[self.classNames[cls]]
                    distance = (real_size * self.focal_length) / object_size_in_frame
                    distance = round(distance, 2)
                    smoothed_distance = self.calculate_moving_avg(self.classNames[cls], distance)

                    objects.append((cls, confidence, smoothed_distance, (x1, y1, x2, y2)))


            # Sort objects by distance (closest first)
            objects.sort(key=lambda obj: obj[2])

            # Prepare text to speak
            text_to_speak = []
            for i, obj in enumerate(objects, 1):
                cls, confidence, distance, coords = obj
                x1, y1, x2, y2 = coords
                direction = self.calculate_direction((x1 + x2) // 2)

                # Create shorter phrases
                label = f"{self.classNames[cls]} at {distance:.2f} meters, {direction}"
                
                print(label) 
                #time.sleep(1)

                text_to_speak.append(label)

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Update or create tracked object
                if cls in self.tracked_objects:
                    self.tracked_objects[cls].coords = coords
                else:
                    self.tracked_objects[cls] = TrackedObject(cls, coords)

            # Get the current time
            current_time = time.time()

            # Speak the accumulated text if the speech interval has passed
            if current_time - self.last_speech_time >= self.speech_interval:
                if text_to_speak:
                    # Submit the TTS task to the thread pool executor for asynchronous execution
                    self.executor.submit(self.speak_text, "\n".join(text_to_speak))
                self.last_speech_time = current_time

            # Analyze the image if the analysis interval has passed
            if current_time - self.last_analysis_time >= self.analysis_interval:
                # Submit the image analysis task to the thread pool executor for asynchronous execution
                self.executor.submit(self.analyze_image, img)
                self.last_analysis_time = current_time

            

            # Update the FPS counter and display the frame
            self.fps.update()
            self.fps.stop()
            fps_value = self.fps.fps()
            cv2.putText(img, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Webcam', img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Approx. FPS: {fps_value:.2f}")
        
    def end_program(self):
        # Cleanup resources
        self.cap.stop()
        cv2.destroyAllWindows()
        self.program_running = False
# Instantiate and run the object detection system
if __name__ == "__main__":
    obj_detection_system = BlindAssistant()
    obj_detection_system.run()

