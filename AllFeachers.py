import cv2
import dlib
import numpy as np
from scipy.signal import medfilt

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Tracking variables
talking_frames = 0
silent_frames = 0
mouth_ratios = []

def detect_talking(landmarks):
    global talking_frames, silent_frames, mouth_ratios

    mouth_left = landmarks.part(48).x
    mouth_right = landmarks.part(54).x
    mouth_top = landmarks.part(51).y
    mouth_bottom = landmarks.part(57).y

    mouth_width = abs(mouth_right - mouth_left)
    mouth_height = abs(mouth_top - mouth_bottom)

    # Mouth aspect ratio calculation
    mouth_ratio = mouth_height / (mouth_width + 0.001)

    # Smooth values with a median filter
    mouth_ratios.append(mouth_ratio)
    if len(mouth_ratios) > 10:
        mouth_ratios.pop(0)

    smooth_ratio = np.median(mouth_ratios)

    # Improved thresholds and stability logic
    if smooth_ratio > 0.28:
        talking_frames += 1
        silent_frames = 0
    elif smooth_ratio < 0.18:
        silent_frames += 1
        talking_frames = 0

    if talking_frames >= 8:
        return "Talking", (0, 165, 255)
    elif silent_frames >= 8:
        return "Silent", (0, 200, 0)
    else:
        return "Uncertain", (255, 255, 0)

# Detect emotions
def detect_emotion(landmarks):
    left_eye_height = abs(landmarks.part(37).y - landmarks.part(41).y)
    right_eye_height = abs(landmarks.part(44).y - landmarks.part(46).y)
    eye_openness = (left_eye_height + right_eye_height) / 2

    mouth_left = landmarks.part(48).x
    mouth_right = landmarks.part(54).x
    mouth_width = abs(mouth_right - mouth_left)

    mouth_top = landmarks.part(51).y
    mouth_bottom = landmarks.part(57).y
    mouth_height = abs(mouth_top - mouth_bottom)

    if mouth_height / mouth_width > 0.3 and eye_openness > 5:
        return "Happy", (0, 255, 0)
    elif mouth_height / mouth_width < 0.1 and eye_openness < 3:
        return "Sad", (255, 0, 0)
    else:
        return "Neutral", (200, 200, 200)

def get_gaze_ratio(eye_points, landmarks, frame):
    left_point = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right_point = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)

    if (left_point[0] < 0 or right_point[0] > frame.shape[1] or
        left_point[1] < 0 or right_point[1] > frame.shape[0]):
        return 1  # Neutral gaze if out of bounds

    eye_region = frame[left_point[1]:right_point[1], left_point[0]:right_point[0]]

    if eye_region.size == 0:
        return 1  # Default to neutral gaze to avoid crashes

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side = threshold_eye[:, 0:int(width / 2)]
    right_side = threshold_eye[:, int(width / 2):]

    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)

    gaze_ratio = left_white / (right_white + 0.001)
    return gaze_ratio

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
        emotion_status, emotion_color = detect_emotion(landmarks)

        left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame)
        right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame)
        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if gaze_ratio < 0.9:
            gaze_direction = "Looking RIGHT"
        elif gaze_ratio > 1.1:
            gaze_direction = "Looking LEFT"
        else:
            gaze_direction = "Looking CENTER"

        if talking_status == "Talking":
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 165, 255), 3)

        cv2.rectangle(frame, (40, 70), (400, 270), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {talking_status}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, talking_color, 2)
        cv2.putText(frame, f"Emotion: {emotion_status}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
        cv2.putText(frame, f"Gaze: {gaze_direction}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    counter += 1
    cv2.imshow("Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
