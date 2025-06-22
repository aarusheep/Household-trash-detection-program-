
from ultralytics import YOLO
import cv2

model = YOLO('yolo_train_project/my_model/weights/best.pt')

img = cv2.imread('C:/Users/aarus/Downloads/MILK PACKET.v1i.yolov8/test/images/milk-packet-4_jpg_jpg.rf.61619d0f2e74edf625f442c660a45655.jpg')
results = model(img)
annotated = results[0].plot()

cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
results = model(img)  # or model(img)
print(results[0].boxes)  # See what boxes were predicted
