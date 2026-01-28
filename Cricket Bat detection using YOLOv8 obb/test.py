from ultralytics import YOLO


model = YOLO("model/weights/best.pt")
model.predict(
    source = "test_sample.mp4",
    project = "results",
    task = "obb",
    conf = 0.25,
    imgsz = 640,
    device = "cpu",
    save = True,
    save_txt = False,
    name = "detection output",
    verbose = False
    )