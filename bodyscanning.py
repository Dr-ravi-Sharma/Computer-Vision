import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
import sys

# COMPLETELY SILENCE ALL WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Initialize MediaPipe with suppressed warnings
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1  # Reduced model complexity for better performance
)

# Constants
REF_WIDTH_CM = 21.0  # A4 paper width
DEFAULT_HEIGHT_CM = 175.0


class BodyMeasurementTool:
    def __init__(self):
        self.px_per_cm = None
        self.measurements = {}
        self.visualization_data = []

    def calculate_measurements(self, landmarks, height_cm=DEFAULT_HEIGHT_CM):
        if not landmarks:
            return None

        try:
            # Keypoint indices
            LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
            RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
            LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
            RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

            # Get landmark positions
            ls = landmarks.landmark[LEFT_SHOULDER]
            rs = landmarks.landmark[RIGHT_SHOULDER]
            lh = landmarks.landmark[LEFT_HIP]
            rh = landmarks.landmark[RIGHT_HIP]

            # Store points for visualization
            self.visualization_data = [
                (ls, rs, "Shoulder", (0, 255, 0)),  # Green
                (lh, rh, "Hip", (0, 0, 255))  # Red
            ]

            # Calculate pixel distances
            shoulder_px = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
            hip_px = np.sqrt((lh.x - rh.x) ** 2 + (lh.y - rh.y) ** 2)

            # Convert to centimeters
            if self.px_per_cm:
                shoulder_cm = shoulder_px / self.px_per_cm
                hip_cm = hip_px / self.px_per_cm
            else:
                shoulder_cm = shoulder_px * height_cm * 0.45
                hip_cm = hip_px * height_cm * 0.45

            self.measurements = {
                "shoulder_cm": round(shoulder_cm, 1),
                "hip_cm": round(hip_cm, 1),
                "height_cm": height_cm,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return self.measurements

        except Exception:
            return None

    def calibrate(self, frame, reference_width_cm=REF_WIDTH_CM):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w_px, h_px = cv2.boundingRect(largest)
                self.px_per_cm = w_px / reference_width_cm
                return True
            return False
        except Exception:
            return False

    def draw_measurements(self, frame):
        try:
            if frame is None or not self.visualization_data:
                return frame

            h, w = frame.shape[:2]
            y_offset = 30
            text_color = (255, 255, 255)  # White

            # Display measurements
            for name, value in self.measurements.items():
                if name.endswith('_cm') and not name == 'height_cm':
                    text = f"{name.replace('_cm', '').title()}: {value}cm"
                    cv2.putText(frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    y_offset += 30

            # Draw measurement lines
            for point1, point2, label, color in self.visualization_data:
                try:
                    pt1 = (int(point1.x * w), int(point1.y * h))
                    pt2 = (int(point2.x * w), int(point2.y * h))

                    if all(0 <= x < w for x in [pt1[0], pt2[0]]) and \
                            all(0 <= y < h for y in [pt1[1], pt2[1]]):
                        cv2.line(frame, pt1, pt2, color, 2)
                        cv2.circle(frame, pt1, 5, color, -1)
                        cv2.circle(frame, pt2, 5, color, -1)

                        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                        cv2.putText(frame, label, (mid_x - 40, mid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except:
                    continue

            return frame

        except Exception:
            return frame

    def save_measurements(self, filename="body_measurements.csv"):
        if not self.measurements:
            return

        try:
            df = pd.DataFrame([self.measurements])
            if os.path.exists(filename):
                existing = pd.read_csv(filename)
                df = pd.concat([existing, df])
            df.to_csv(filename, index=False)
        except Exception:
            pass


def initialize_camera():
    """Initialize camera with platform-specific backend"""
    backends = [
        cv2.CAP_AVFOUNDATION,  # macOS
        cv2.CAP_DSHOW,  # Windows
        cv2.CAP_V4L2  # Linux
    ]

    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None


def main():
    tool = BodyMeasurementTool()
    cap = None

    try:
        # Get user height
        try:
            height = float(input(f"Enter height in cm (default {DEFAULT_HEIGHT_CM}): ") or DEFAULT_HEIGHT_CM)
        except (ValueError, KeyboardInterrupt):
            height = DEFAULT_HEIGHT_CM

        # Initialize camera
        cap = initialize_camera()
        if not cap:
            print("Error: Could not initialize camera")
            return

        print("\nControls:")
        print("'c' - Calibrate with A4 paper")
        print("'m' - Take measurements")
        print("'q' - Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read error")
                break

            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if tool.calibrate(frame):
                    print("Calibration successful")
                else:
                    print("Calibration failed - show reference object")
            elif key == ord('m'):
                if results.pose_landmarks:
                    tool.calculate_measurements(results.pose_landmarks, height)
                    tool.save_measurements()
                    print("Measurements saved")
            elif key == ord('q'):
                break

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame = tool.draw_measurements(frame)

            status = "CALIBRATED" if tool.px_per_cm else "UNCALIBRATED"
            color = (0, 255, 0) if tool.px_per_cm else (0, 0, 255)
            cv2.putText(frame, status, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Body Measurement Tool', frame)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Ensure camera is properly released
        if cap and cap.isOpened():
            cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        # Additional cleanup for MacOS
        if os.name == 'posix':
            os.system('pkill -f "VDCAssistant"')
            os.system('pkill -f "AppleCameraAssistant"')


if __name__ == "__main__":
    # Final suppression of MediaPipe/TensorFlow logs
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    main()