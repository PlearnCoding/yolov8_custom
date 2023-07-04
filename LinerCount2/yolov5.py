import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

def loadWeight():
    # weights = 'weights/best_v5x_83.3_defect06.pt'
    weights = 'weights/best_cylinder_n2.pt'
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    return model

def cornerRectangle(x, img, colorB=(0,255,255),colorR=(0,255,0),factor=0.12, thicknessB=1,thicknessR=2):
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

def roundCorner(x, img, colorB=(0,255,255),colorR=(0,255,0), factor=0.12, thicknessB=1,thicknessR=2):
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

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    factor=0.12
    l = int(min(c2[0]-c1[0],c2[1]-c1[1])*factor)
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # cornerRectangle(x, img, colorB=color,colorR=(0,255,255), factor=factor, thicknessB=line_thickness,thicknessR=3)
    roundCorner(x, img, colorB=color,colorR=(0,255,0), factor=factor, thicknessB=line_thickness,thicknessR=line_thickness+1)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = c1[0] + l ,c1[1]
        c2 = c1[0]+ t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA) #[225, 255, 255]

def detect(model,img_source,img_size,conf_thres,iou_thres,debug=False,rmin=100,rmax=200):

    set_logging()
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = check_img_size(img_size,32 )  # s=model.stride.max() check img_size

    icam = img_source[-5:-4]
    defect = 0
    save_img = True
    dataset = LoadImages(img_source, img_size=imgsz)
    n_defect = {}
    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    names = ['Cyl']
    random.seed(101)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0,255,255]]
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=False)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    print('detect :',s)
                    n_defect[names[int(c)]] = n #n.cpu().numpy()
                    #defect +=n
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_img :#or view_img:  # Add bbox to image
                    if debug:
                        label = '%s %.2f' % (names[int(cls)], conf)
                    else :
                        label = '%s' % (names[int(cls)])
                    x1,y1 = int(xyxy[0]), int(xyxy[1])
                    x2,y2 = int(xyxy[2]), int(xyxy[3])
                    l,u = min(x2-x1,y2-y1),max(x2-x1,y2-y1)
                    if l>rmin and u<rmax :
                        print(f'Dia : {l} pixels ,Area : {(x2-x1)*(y2-y1)} pixels2')
                        defect +=1
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            timestr01 = time.strftime('%Y%m%d-%H%M%S')
            #if defect > 0 :
                #cv2.imwrite('per/images_result/cam_0'+str(icam)+'/result_cam_capture0'+str(icam)+'_'+timestr01+'.jpg',im0)
            cv2.imwrite('temp/images_result/result_cam_0'+str(icam)+'.jpg',im0)
            # print('save image')

    # print('Done. (%.3fs)' % (time.time() - t0))
    print('Total defect = :',defect)
    return defect,n_defect,im0

'''
if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--weights', nargs='+', type=str, default='weights/best_yolov5l_81.1_defect06.pt', help='model.pt path(s)')
     parser.add_argument('--source', type=str, default='ImagesAttendance/Ritz.png', help='source')  # file/folder, 0 for webcam
     parser.add_argument('--output', type=str, default='inference/', help='output folder')  # output folder
     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
     parser.add_argument('--view-img', action='store_true', help='display results')
     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
     parser.add_argument('--augment', action='store_true', help='augmented inference')
     parser.add_argument('--update', action='store_true', help='update all models')
     opt = parser.parse_args()
     print(opt)

     with torch.no_grad():
         if opt.update:  # update all models (to fix SourceChangeWarning)
             for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                 detect()
                 strip_optimizer(opt.weights)
         else:
    img_source = ''
    model = loadWeight()
    detect(model,img_source,640,0.5,0.4)
'''
