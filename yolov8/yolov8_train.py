from ultralytics import YOLO

# Load the model.

if __name__ == "__main__" :

    model = YOLO('weights/yolov8s.pt')
    
    # Training.
    results = model.train(
    data='knuckle2.yaml',
    imgsz=640,
    epochs=50,
    batch=4,
    workers=1,
    name='kn2_v8s_50e'
    )

