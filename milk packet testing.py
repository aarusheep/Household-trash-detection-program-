from ultralytics import YOLO

# Load a YOLOv8 model (you can start from 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', or 'yolov8x')
model = YOLO('yolov8n.pt')  # use 'yolov8s.pt', etc., for larger models

# Train the model
model.train(
    data='C:/Users/aarus/Downloads/MILK PACKET.v1i.yolov8/data.yaml',

    epochs=50,
    imgsz=640,
    batch=16,
    project='yolo_train_project',
    name='my_model',
    exist_ok=True
)
