import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import time
import threading
from queue import Queue
import asyncio
import aiohttp
import cvzone
import tempfile
from openai import OpenAI

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Accident Detector", layout="wide")

# Initialize OpenAI client 
DEFAULT_API_KEY = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEFAULT_API_KEY,
)

class CarAccidentDetectionProcessor:
    def __init__(self, video_path, api_key, model_path="yolo12s.pt"):
        """Initialize car accident detection processor."""
        self.api_key = api_key
        self.node_red_url = "http://localhost:1880/accident-alert"  # Node-RED Endpoint
        
        # Load Model
        try:
            self.yolo_model = YOLO(model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            st.stop()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            st.error("Error: Could not open video file.")
            st.stop()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Setup folders
        self.current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.images_folder = os.path.join("sent_images", f"images_{self.current_date}")
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
        
        self.output_filename = f"accident_data_{self.current_date}.txt"
        
        # Runtime variables
        self.image_counter = 0
        self.frame_counter = 0
        self.last_frame_sent_time = 0
        self.frame_queue = Queue(maxsize=10)
        self.analysis_queue = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_response = "Waiting for analysis..."
        
        # File Init
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Accident Severity | Vehicles Involved | Location Details | Reason\n")
                file.write("-" * 100 + "\n")

    # --- NODE-RED SENDER ---
    async def send_to_nodered(self, data):
        """Sends accident data to Node-RED webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.node_red_url, json=data) as response:
                    if response.status == 200:
                        print(f"✅ Alert sent to Node-RED for Car ID: {data['car_id']}")
        except Exception as e:
            print(f"❌ Failed to send to Node-RED: {e}")

    # --- OPENAI ANALYSIS ---
    async def analyze_frame_with_openai(self, frame, track_id, session):
        try:
            # Resize for speed/cost
            frame_small = cv2.resize(frame, (320, 240))
            _, img_buffer = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            base64_image = base64.b64encode(img_buffer).decode("utf-8")

            prompt = """
            Analyze this CCTV frame for car accidents. Focus on:
            - Vehicle Damage, Unusual Positions, Debris, Traffic Disruption.
            
            Output strictly a single line separated by pipes (|) like this:
            Accident Severity | Vehicles Involved | Location Type | Likely Cause
            
            Example:
            Medium | 2 cars | Intersection | Side collision
            """
            
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "http://localhost",
                },
                json={
                    "model": "qwen/qwen-2.5-vl-7b-instruct:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                }
            ) as response:
                result = await response.json()
                if response.status == 200 and "choices" in result:
                    return result["choices"][0]["message"]["content"].strip()
                return "Error: API response invalid"
        except Exception as e:
            return f"Error processing image: {e}"

    # --- BACKGROUND THREAD FOR AI ---
    def process_frame_in_thread(self):
        async def run_analysis():
            async with aiohttp.ClientSession() as session:
                while not self.stop_event.is_set():
                    if self.analysis_queue:
                        with self.lock:
                            frame, timestamp, track_id = self.analysis_queue.pop(0)
                        
                        response_content = await self.analyze_frame_with_openai(frame, track_id, session)
                        
                        # Process response if it follows the pipe format
                        if "|" in response_content:
                            try:
                                parts = [p.strip() for p in response_content.split('|')]
                                if len(parts) >= 4:
                                    severity = parts[0]
                                    vehicles = parts[1]
                                    location = parts[2]
                                    cause = parts[3]
                                    
                                    # Update GUI text
                                    self.latest_response = f"TIMESTAMP: {timestamp}\nID: {track_id}\nSEVERITY: {severity}\nDETAILS: {cause}"
                                    
                                    # Log to file
                                    with open(self.output_filename, "a", encoding="utf-8") as file:
                                        file.write(f"{timestamp} | {track_id} | {response_content}\n")

                                    # Trigger Node-RED if Medium or High Severity
                                    if "Medium" in severity or "High" in severity:
                                        alert_data = {
                                            "timestamp": timestamp,
                                            "car_id": track_id,
                                            "severity": severity,
                                            "location": location,
                                            "cause": cause,
                                            "message": f"🚨 ACCIDENT DETECTED 🚨\n\nSeverity: {severity}\nCar ID: {track_id}\nLocation: {location}\nCause: {cause}"
                                        }
                                        # Send async task
                                        asyncio.create_task(self.send_to_nodered(alert_data))
                            except Exception as parse_error:
                                print(f"Parsing error: {parse_error}")
                        else:
                            print(f"AI Response (No Accident format): {response_content}")

                    else:
                        await asyncio.sleep(0.1)
        
        # Run async loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_analysis())
        loop.close()

    def process_queue_request(self, frame, track_id):
        current_time = time.time()
        # Analyze every 5 seconds to avoid spamming API
        if current_time - self.last_frame_sent_time >= 5:
            self.last_frame_sent_time = current_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with self.lock:
                if len(self.analysis_queue) < 10:
                    self.analysis_queue.append((frame.copy(), timestamp, track_id))

    def process_frame(self, frame):
        # YOLO Tracking
        frame_resized = cv2.resize(frame, (640, 480))
        results = self.yolo_model.track(frame_resized, persist=True, verbose=False)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            scale_x, scale_y = self.frame_width / 640, self.frame_height / 480
            
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = self.names[class_id]
                
                # Check for cars or accidents
                if class_name in ["car", "accident", "truck", "bus"]:
                    x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                    
                    # Draw Box
                    color = (0, 0, 255) if class_name == "accident" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Label
                    cvzone.putTextRect(frame, f"{class_name} {track_id}", (x1, y1 - 10), scale=1, thickness=2, offset=5)
                    
                    # Send for Analysis
                    self.process_queue_request(frame, track_id)

        return frame

    def read_frames(self):
        while self.cap.isOpened() and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_queue.put(frame)
        self.frame_queue.put(None)

    def start_streamlit_processing(self, st_image_placeholder, st_text_placeholder):
        # Start Threads
        t_read = threading.Thread(target=self.read_frames, daemon=True)
        t_analyze = threading.Thread(target=self.process_frame_in_thread, daemon=True)
        t_read.start()
        t_analyze.start()

        stop_button = st.button("Stop Processing")

        while True:
            if stop_button:
                self.stop_event.set()
                break

            try:
                # Non-blocking get
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    if frame is None:
                        break # End of video
                    
                    self.frame_counter += 1
                    
                    # Process every 3rd frame to keep up with UI
                    if self.frame_counter % 3 == 0:
                        processed_frame = self.process_frame(frame)
                        
                        # Add Overlay for Latest AI Response
                        cvzone.putTextRect(processed_frame, "AI Analysis:", (10, 30), scale=1, thickness=2, colorR=(0,0,0))
                        y_offset = 60
                        for line in self.latest_response.split('\n'):
                            cv2.putText(processed_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                            y_offset += 25
                        
                        # Convert BGR to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st_image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Use .code() to avoid "Duplicate ID" errors
                        st_text_placeholder.code(self.latest_response, language="yaml")
                
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                st.error(f"Stream Error: {e}")
                break

        self.stop_event.set()
        self.cap.release()
        st.success("Processing Complete!")

# --- Main Streamlit Interface ---

def main():
    st.title("🚗 AI Car Accident Detection System")
    st.markdown("Upload a video to detect accidents. Alerts are sent to **Node-RED** automatically.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input("OpenRouter API Key", value=DEFAULT_API_KEY, type="password")
        model_path = st.text_input("YOLO Model Path", value="yolo12s.pt")
        st.info("Ensure Node-RED is running on port 1880.")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st_image_placeholder = st.empty()
        with col2:
            st.markdown("### 📋 Live Analysis")
            st_text_placeholder = st.empty()

        if st.button("▶️ Start Detection"):
            processor = CarAccidentDetectionProcessor(tfile.name, api_key_input, model_path)
            processor.start_streamlit_processing(st_image_placeholder, st_text_placeholder)

if __name__ == "__main__":
    main()