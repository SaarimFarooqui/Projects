from ultralytics import YOLO

model = YOLO("yolov8s-obb.pt")

model.train(
    task = "obb",
    data = "dataset\data.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 16,
    device = 0,
    name = "bestBat_OBB",
    project = "model",
    save = True 
    )