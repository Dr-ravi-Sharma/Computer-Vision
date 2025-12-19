from facenet_pytorch import MTCNN
import cv2

# Initialize MTCNN
mtcnn = MTCNN()

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Detect Faces
    boxes, _ = mtcnn.detect(frame)

    # Draw Bounding Boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display Frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()