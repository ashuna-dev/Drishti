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

class BlindAssistant:
    def __init__(self, src='rtsp://192.168.0.100:8080/h264_ulaw.sdp'):
        # Initialize video stream and YOLO model
        self.src = src
        self.cap = WebcamVideoStream(src).start()
        self.model = YOLO("yolov8n.pt")
        self.fps = FPS().start()

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
        
        
        self.object_real_size = {"person": 5.5, "bicycle": 6.0, "car": 15.0, "motorbike": 6.0, "aeroplane": 30.0, 
                                  "bus": 40.0, "train": 60.0, "truck": 25.0, "boat": 20.0, "traffic light": 3.0, 
                                  "fire hydrant": 3.0, "stop sign": 3.0, "parking meter": 4.0, "bench": 5.0, 
                                  "bird": 0.5, "cat": 1.0, "dog": 1.0, "horse": 8.0, "sheep": 5.0, "cow": 6.0, 
                                  "elephant": 12.0, "bear": 7.0, "zebra": 7.0, "giraffe": 16.0, "backpack": 1.5, 
                                  "umbrella": 2.0, "handbag": 1.5, "tie": 1.0, "suitcase": 3.0, "frisbee": 1.0, 
                                  "skis": 5.0, "snowboard": 5.0, "sports ball": 1.0, "kite": 3.0, "baseball bat": 3.0,
                                  "baseball glove": 1.0, "skateboard": 3.0, "surfboard": 6.0, "tennis racket": 3.0,
                                  "bottle": 1.0, "wine glass": 1.0, "cup": 3.0, "fork": 1.0, "knife": 1.0, 
                                  "spoon": 1.0, "bowl": 4.0, "banana": 1.0, "apple": 1.0, "sandwich": 4.0, 
                                  "orange": 2.0, "broccoli": 2.0, "carrot": 2.0, "hot dog": 2.0, "pizza": 5.0, 
                                  "donut": 1.0, "cake": 5.0, "chair": 3.0, "sofa": 6.0, "potted plant": 2.0, 
                                  "bed": 6.0, "dining table": 6.0, "toilet": 3.0, "tv monitor": 5.0, "laptop": 1.0, 
                                  "mouse": 1.0, "remote": 1.0, "keyboard": 1.0, "cell phone": 1.0, "microwave": 3.0, 
                                  "oven": 3.0, "toaster": 2.0, "sink": 2.0, "refrigerator": 5.0, "book": 1.0, 
                                  "clock": 1.0, "vase": 2.0, "scissors": 1.0, "teddy bear": 2.0, "hair drier": 1.0, 
                                  "toothbrush": 1.0}
        
        self.focal_length = 558  # Focal length (estimated, you may need to adjust)

        # Define intervals
        self.speech_interval = 5
        self.analysis_interval = 5

        # Initialize last speech and analysis times
        self.last_speech_time = time.time()
        self.last_analysis_time = time.time()

        # Configure Google Generative AI
        self.api_key = 'AIzaSyAHR1jdjWT1CF3rGoNhyRmJbEWjbi5GMMw'  # Replace with your API key
        genai.configure(api_key=self.api_key)
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048
        }
        self.model_genai = genai.GenerativeModel("gemini-pro-vision", generation_config=generation_config)

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=1)

    def speak_text(text):
     # Create the text-to-speech audio file
        tts = gTTS(text=text, lang="en")
        tts_path = "temp.mp3"
        tts.save(tts_path)

        # Use wmic to invoke Windows Media Player in hidden mode and play the audio file
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
        prompt_parts = ["describe image and ignore boxes and text over it, also any fps", image_part]
        try:
            response = self.model_genai.generate_content(prompt_parts)

            # Check the response for correctness
            if response and response.text:
                # Print the response text
                print("Analysis response:", response.text)

                # Speak the response text
                self.speak_text(response.text)
            else:
                print("No valid response received from generative AI.")
        except Exception as e:
            print(f"Error during analysis: {e}")

        # Clean up the temporary image file
        os.remove(image_path)

    

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
                    real_size = self.object_real_size[self.classNames[cls]]
                    distance = (real_size * self.focal_length) / object_size_in_frame
                    distance = round(distance, 2)
                    objects.append((cls, confidence, distance, (x1, y1, x2, y2)))

            # Sort objects by distance (closest first)
            objects.sort(key=lambda obj: obj[2])

            # Prepare text to speak
            text_to_speak = []
            for i, obj in enumerate(objects, 1):
                cls, confidence, distance, coords = obj
                x1, y1, x2, y2 = coords
                # Create shorter phrases
                label = f"{self.classNames[cls]} at {distance:.2f} ft"
                print(label)  # Output to command line
                text_to_speak.append(label)

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

            # Perform additional frame processing
            

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

# Instantiate and run the object detection system
if __name__ == "__main__":
    obj_detection_system = BlindAssistant()
    obj_detection_system.run()
