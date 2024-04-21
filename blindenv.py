import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import cv2
import math
import google.generativeai as genai
from imutils.video import WebcamVideoStream, FPS
from gtts import gTTS
from pathlib import Path
import os

# Configure Google Generative AI
api_key = 'AIzaSyAHR1jdjWT1CF3rGoNhyRmJbEWjbi5GMMw'  # Replace 'your_api_key_here' with your API key
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}
model_genai = genai.GenerativeModel("gemini-pro-vision", generation_config=generation_config)


# Initialize the video stream and YOLO model
src="rtsp://192.168.29.8:8080/h264_ulaw.sdp"
cap = WebcamVideoStream(src=0).start()
model = YOLO("yolov8n.pt")
fps = FPS().start()

# Define object classes and focal length for distance calculation
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

object_real_size = 5.5  # Approximate real size of a person in feet
focal_length = 558  # Focal length (estimated, you may need to adjust)

# Define the speech interval and image analysis interval
speech_interval = 4
analysis_interval = 7

last_speech_time = time.time()
last_analysis_time = time.time()

# Function to use macOS say command to convert text to speech
def speak_text(text):
     # Create the text-to-speech audio file
    tts = gTTS(text=text, lang="en")
    tts_path = "temp.mp3"
    tts.save(tts_path)

        # Use wmic to invoke Windows Media Player in hidden mode and play the audio file
    subprocess.run(["wmic", "process", "call", "create", '"wmplayer.exe /play /close ' + tts_path + '"'])

# Function to analyze the image using google.generativeai
def analyze_image(img):
    # Save the image temporarily as 'image.jpg' for analysis
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
        response = model_genai.generate_content(prompt_parts)

        # Check the response for correctness
        if response and response.text:
            # Print the response text
            print("Analysis response:", response.text)
            
            # Speak the response text
            speak_text(response.text)
        else:
            print("No valid response received from generative AI.")
    except Exception as e:
        print(f"Error during analysis: {e}")

    # Clean up the temporary image file
    os.remove(image_path)

# Create a thread pool executor with a maximum of one worker (for sequential TTS)
with ThreadPoolExecutor(max_workers=1) as executor:
    # Main loop
    while True:
        img = cap.read()
        if img is None or img.size == 0:
            continue  # Skip invalid frames

        results = model(img, stream=True)

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
                distance = (object_real_size * focal_length) / object_size_in_frame
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
            label = f"{classNames[cls]} at {distance:.2f} ft"
            print(label)
            time.sleep(1)
            text_to_speak.append(label)

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Get the current time
        current_time = time.time()

        # Speak the accumulated text if the speech interval has passed
        if current_time - last_speech_time >= speech_interval:
            # Submit the TTS task to the thread pool executor for asynchronous execution
            if text_to_speak:
                executor.submit(speak_text, "\n".join(text_to_speak))
            # Update the last speech time
            last_speech_time = current_time

        # Analyze the image if the analysis interval has passed
        if current_time - last_analysis_time >= analysis_interval:
            # Analyze the image using the analyze_image function
            executor.submit(analyze_image, img)
            # Update the last analysis time
            last_analysis_time = current_time

        # Update the FPS counter and display the frame
        fps.update()
        fps.stop()
        fps_value = fps.fps()
        cv2.putText(img, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Webcam', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.stop()
cv2.destroyAllWindows()
print(f"[INFO] Approx. FPS: {fps_value:.2f}")

