import cv2
import dlib
import numpy as np
from scipy.signal import medfilt

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Mouth movement tracking
talking_frames = 0
silent_frames = 0
mouth_ratios = []

# Detect talking state
def detect_talking(landmarks):
    global talking_frames, silent_frames, mouth_ratios

    mouth_left = landmarks.part(48).x
    mouth_right = landmarks.part(54).x
    mouth_top = landmarks.part(51).y
    mouth_bottom = landmarks.part(57).y

    mouth_width = abs(mouth_right - mouth_left)
    mouth_height = abs(mouth_top - mouth_bottom)

    # Mouth aspect ratio calculation
    mouth_ratio = mouth_height / (mouth_width + 0.001)  # Avoid division by zero

    # Smooth values with a median filter to reduce noise
    mouth_ratios.append(mouth_ratio)
    if len(mouth_ratios) > 10:
        mouth_ratios.pop(0)

    smooth_ratio = np.median(mouth_ratios)

    # Improved thresholds and stability logic
    if smooth_ratio > 0.28:  # Talking threshold
        talking_frames += 1
        silent_frames = 0
    elif smooth_ratio < 0.18:  # Silent threshold
        silent_frames += 1
        talking_frames = 0

    if talking_frames >= 8:  # Stable condition for talking
        return "Talking", (0, 165, 255)
    elif silent_frames >= 8:  # Stable condition for silent
        return "Silent", (0, 200, 0)
    else:
        return "Uncertain", (255, 255, 0)

# Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

frame_skip = 5
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if counter % frame_skip == 0:
        faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        talking_status, talking_color = detect_talking(landmarks)

        cv2.rectangle(frame, (40, 70), (400, 270), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {talking_status}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, talking_color, 2)

    counter += 1
    cv2.imshow("Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
