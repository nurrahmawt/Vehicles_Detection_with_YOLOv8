from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

cap = cv2.VideoCapture("video_traffic.mp4")  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("Vehicle Detection Demo", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()