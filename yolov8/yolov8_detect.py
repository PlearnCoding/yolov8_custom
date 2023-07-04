from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("weights/yolov8s.pt")
# results = model.predict(source="https://ultralytics.com/images/bus.jpg",imgsz=[640,640])[0]
# Define yolov8 classes
CLASESS_YOLO = [
 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 
 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 
 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 
 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
def yolov8_detect(model, image_path, conf=0.35):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, imgsz=[640,640])[0]  # generator of Results objects

    boxes_list = []
    classes_list = []
    conf_list = []

    for result in results.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = result[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < conf:
            continue
        boxes_list.append([int(result[0]), int(result[1]), int(result[2]), int(result[3])])
        conf_list.append(float(result[4]))
        classes_list.append(int(result[5]))
        

    return boxes_list, conf_list, classes_list, image

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    # Create a list of colors for each class where each color is a tuple of 3 integer values
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(CLASESS_YOLO), 3))

    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = CLASESS_YOLO[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

def cornerRectangle(x, img, colorB=(0,255,255),colorR=(0,255,0), factor=0.15, thicknessB=1,thicknessR=2):
    x1,y1 = int(x[0]), int(x[1])
    x2,y2 = int(x[2]), int(x[3])
    l = int(min(x2-x1,y2-y1)*factor)
    cv2.rectangle(img,(x1,y1),(x2,y2),colorB,thicknessB)
    cv2.line(img, (x1,y1), (x1+l,y1), colorR, thicknessR,0)
    cv2.line(img, (x1,y1), (x1,y1+l), colorR, thicknessR,0)
    cv2.line(img, (x2,y2), (x2-l,y2), colorR, thicknessR,0)
    cv2.line(img, (x2,y2), (x2,y2-l), colorR, thicknessR,0)
    cv2.line(img, (x2,y1), (x2-l,y1), colorR, thicknessR,0)
    cv2.line(img, (x2,y1), (x2,y1+l), colorR, thicknessR,0)
    cv2.line(img, (x1,y2), (x1+l,y2), colorR, thicknessR,0)
    cv2.line(img, (x1,y2), (x1,y2-l), colorR, thicknessR,0)

def roundCorner(x, img, colorB=(0,255,255),colorR=(0,255,0), factor=0.15, thicknessB=1,thicknessR=2):
    x1,y1 = int(x[0]), int(x[1])
    x2,y2 = int(x[2]), int(x[3])
    l = int(min(x2-x1,y2-y1)*factor)
    cv2.ellipse(img,(x1+l,y1+l),(l,l),0,-180,-90,colorR,thicknessR,0)
    cv2.ellipse(img,(x2-l,y2-l),(l,l),0,0,90,colorR,thicknessR,0)
    cv2.ellipse(img,(x2-l,y1+l),(l,l),0,-90,0,colorR,thicknessR,0)
    cv2.ellipse(img,(x1+l,y2-l),(l,l),0,-180,-270,colorR,thicknessR,0)
    cv2.line(img, (x1+l,y1), (x2-l,y1), colorB, thicknessB,0)
    cv2.line(img, (x1+l,y2), (x2-l,y2), colorB, thicknessB,0)
    cv2.line(img, (x1,y1+l), (x1,y2-l), colorB, thicknessB,0)
    cv2.line(img, (x2,y1+l), (x2,y2-l), colorB, thicknessB,0)

v8bbox, v8conf, v8class ,img = yolov8_detect(model,'bus.jpg')

# for i in range(len(v8bbox)):
#     xmin, ymin, xmax, ymax = v8bbox[i]
#     conf = v8conf[i]
#     id = v8class[i]
#     # cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(255,255,0),2)
#     roundCorner(v8bbox[i],img)
#     cv2.putText(img,f'{CLASESS_YOLO[id]}:{conf:.2f}',(xmin,ymin-3),0,0.65,(255,255,255),2)

img = draw_detections(img,v8bbox,v8conf,v8class)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", img)
cv2.imwrite("detected_objects.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# model.export(format="onnx",imgsz=[640,640],opset=12)  # export the model to ONNX format
