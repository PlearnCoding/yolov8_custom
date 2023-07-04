from ultralytics import YOLO

model = YOLO("D:/MyCoding/yolov8/runs/detect/kn2_v8s_50e/weights/best_nk2_v8s.995.pt")

model.export(format="onnx",imgsz=[640,640],opset=12)  # export the model to ONNX format