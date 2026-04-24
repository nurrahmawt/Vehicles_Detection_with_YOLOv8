from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

url = "http://192.168.xxx.xxx:8080/video" # import url from IP Webcam app
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()