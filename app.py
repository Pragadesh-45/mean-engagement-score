from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import torch
import numpy as np
import tempfile
import asyncio
import base64
import mediapipe as mp
import os

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Initialize MediaPipe Face Mesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Nose tip landmark index
NOSE_TIP_INDEX = 1

def calculate_attention_score(frame, detections):
    """
    Calculate attention score based on head pose detection.
    """
    attention_scores = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        # Extract face region
        face = frame[int(y1):int(y2), int(x1):int(x2)]
        if face.size == 0:
            continue

        # Perform head pose detection using MediaPipe Face Mesh
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(face_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Simplified attention logic: check if nose tip points forward
                nose_tip = face_landmarks.landmark[NOSE_TIP_INDEX]
                if nose_tip.z < -0.1:  # Assuming negative Z indicates forward direction
                    attention_scores.append(1.0)  # Fully attentive
                else:
                    attention_scores.append(0.0)  # Not attentive

    return np.mean(attention_scores) if attention_scores else 0

def process_frame(frame):
    """
    Process a single frame for attention detection
    """
    # Detect students (persons)
    results = model(frame)
    detections = results.xyxy[0].numpy()

    # Filter by class (assuming 'person' is class 0 in COCO dataset)
    students = [det for det in detections if int(det[5]) == 0]

    # Calculate attention scores for each student
    individual_attention_scores = []
    for det in students:
        x1, y1, x2, y2, conf, cls = det

        # Extract face region
        face = frame[int(y1):int(y2), int(x1):int(x2)]
        if face.size == 0:
            continue

        # Perform head pose detection using MediaPipe Face Mesh
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(face_rgb)

        attention_score = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_tip = face_landmarks.landmark[NOSE_TIP_INDEX]
                # Assuming negative Z indicates forward direction (attentive)
                if nose_tip.z < -0.1:
                    attention_score = 1.0  # Fully attentive
                else:
                    attention_score = 0.0  # Not attentive

        individual_attention_scores.append(attention_score)

        # Annotate the frame with attention score in red
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'Attention: {attention_score * 100:.2f}%', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red color for attention score

    # Calculate the average attention score
    avg_attention = np.mean(individual_attention_scores) if individual_attention_scores else 0

    # Annotate overall attention at the top-left corner
    cv2.putText(frame, f'Overall Attention: {avg_attention * 100:.2f}%', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color for overall attention score

    return frame, avg_attention

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing and streaming
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 encoded frame from client
            data = await websocket.receive_text()
            
            # Decode base64 image
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            processed_frame, attention_score = process_frame(frame)
            
            # Encode processed frame back to base64
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send processed frame and attention score back to client
            await websocket.send_json({
                'frame': f'data:image/jpeg;base64,{processed_frame_b64}',
                'attention_score': f'{attention_score * 100:.2f}%'
            })
    
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload endpoint for processing and saving the entire video
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_path = temp_video.name

    # Create output directories
    os.makedirs("static/frames", exist_ok=True)
    os.makedirs("static/processed", exist_ok=True)

    # Process the entire video (similar to previous implementation)
    cap = cv2.VideoCapture(temp_path)
    attention_percentages = []
    output_frames = []

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, attention_score = process_frame(frame)
        attention_percentages.append(attention_score)
        output_frames.append(processed_frame)

        # Save frame
        frame_path = os.path.join("static/frames", f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, processed_frame)
        frame_index += 1

    cap.release()

    # Calculate overall attention
    overall_attention = np.mean(attention_percentages) * 100

    # Save processed video
    height, width, _ = output_frames[0].shape
    output_video_path = os.path.join("static/processed", "processed_video.mp4")
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for frame in output_frames:
        writer.write(frame)
    writer.release()

    # Return processing results
    frame_urls = [
        f"/static/frames/{filename}" for filename in sorted(os.listdir("static/frames"))
    ]

    return {
        "overall_attention": f"{overall_attention:.2f}%",
        "video_url": f"/static/processed/{os.path.basename(output_video_path)}",
        "frame_urls": frame_urls
    }