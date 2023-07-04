import sys,os ,shutil
import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import imutils
from imutils.video import VideoStream
import yolov5
from models.experimental import attempt_load
from utils.torch_utils import select_device
import time
import glob
from threading import Thread

timestr01 = time.strftime('%Y%m%d-%H%M%S')
inference = [0]

class camera(QMainWindow): #เป็น class ลูกของ QMainWindow
    def __init__(self):
        super(camera, self).__init__()
        loadUi('qtdesigner_detect_1.ui', self)  
        models = ['model_yolo5']
        weights = ['weights/best_09.pt','weights/best_091.pt','weights/best_092.pt']
        print(models)
        print(weights) 
        # model_efficientdet_keras
        self.model_type = 'model_yolo5'
        self.weights = 'weights/yolov5s.pt'
        self.frame = None
        self.cap = []
        self.N = 1   
        self.defect = [0]*self.N
        self.logic = [0]*self.N
        self.value = [0]*self.N
        self.detect_finish = [0]*self.N
        self.recordT = time.time()
        self.det = False
        #self.th = Thread(target=self.detect,args=[0])
        self.Connect.clicked.connect(self.open_cameras_clicked) 
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.setInterval(25)
        self.timer_camera.timeout.connect(self.show_index_camera)

        self.timer_capt = QtCore.QTimer()
        self.timer_capt.setInterval(3000)
        self.timer_capt.timeout.connect(self.detect_index)
        
        self.Exit.clicked.connect(self.ExitClicked)
        self.actionExit.triggered.connect(self.ExitClicked)
        self.actionWeights.triggered.connect(self.ExitClicked)

        self.comboBox_models.setItemText(0,models[0])
        # self.comboBox_models.currentIndexChanged.connect(self.Models_Selectionchanged_1)
        # self.comboBox_models.setEditable(True)
        # self.comboBox_models.addItems(models)
        self.comboBox_models.activated.connect(self.Models_Selectionchanged_1)
        # self.comboBox_weights.setEditable(True)
        # self.comboBox_weights.addItems(weights)
        self.comboBox_weights.setItemText(0,weights[0])
        self.comboBox_weights.setItemText(1,weights[1])
        self.comboBox_weights.setItemText(2,weights[2])
        self.comboBox_weights.activated.connect(self.Weights_Selectionchanged_1)
        # self.comboBox_weights.setItemText(0,weights[0])
        # self.comboBox_weights.setItemText(1,weights[1])
        # self.comboBox_weights.setItemText(2,weights[2])
        # self.comboBox_weights.currentIndexChanged.connect(self.Weights_Selectionchanged_1)

        self.Detect.clicked.connect(self.on_capture)        

    def on_final(self):        
        # print(self.detect_finish_1, self.detect_finish_2, self.detect_finish_3, self.detect_finish_4)
        global inference
        # if self.detect_finish_1==1 or self.detect_finish_2==1 or self.detect_finish_3==1 or self.detect_finish_4==1:
            
        if sum(inference) == 0:
            self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/green.jpg"))
            self.FinalResult_1.setScaledContents(True)                
        else:
            self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
            self.FinalResult_1.setScaledContents(True) 

    def load_weight(self):
        # weights = 'weights/best_v5x_83.3_defect06.pt'
        weights = self.weights
        device = select_device('cpu')
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if half:
            model.half()  # to FP16

        return model

    def open_cameras_clicked(self):
        self.cap =[]
        for i in range(self.N):
            # temp = cv2.VideoCapture(i,cv2.CAP_DSHOW)
            temp = cv2.VideoCapture(i)
            temp.set(3,640); temp.set(4,480)
            if temp.isOpened() == True:
                self.cap.append(temp)
                print('open camera :',i)
                #break
                # print(i)
            else:
                temp.release()
        print('click open camera')
        self.timer_camera.start()
        self.model = self.load_weight()
        self.timer_capt.start()
        #self.th.start()

    def displayImage(self,displayI,img, windown=1):
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image = qimage2ndarray.array2qimage(frame)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            displayI.setPixmap(pixmap)
            displayI.setScaledContents(True)
            # self.camera_display1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def show_camera_thread(self,Icamera): 
        # len_cap = len(self.cap)
        displayI = [self.camera_display1]
        # if len_cap != 0:
        #     for i in range(len_cap):   
        
        ret, self.frame = self.cap[Icamera].read()
        if not ret:
            print("Failed to read Camera No. %d" % (Icamera))
        else:
            self.displayImage(displayI[Icamera],self.frame,1)
            # cv2.waitKey()
            tic = time.time()
            if tic - self.recordT >= 3.0:
                self.on_capture()
                cv2.imwrite(f"temp/camera_capture/cam_capture0{Icamera+1}.jpg",self.frame)
                self.recordT = time.time()
                #self.detect(Icamera)
            '''    
            if (self.logic[Icamera]==1):
                self.value[Icamera] = self.value[Icamera]+1
                cv2.imwrite(f"temp/camera_capture/cam_capture0{Icamera+1}.jpg",self.frame)
                timestr01 = time.strftime('%Y%m%d-%H%M%S')
                # cv2.imwrite(f'per/camera_capture/cam_0{Icamera+1}/cam_capture0{Icamera+1}_{timestr01}.jpg',frame)
                #self.logic[Icamera]=0 # Afterer imwrite, will clear logic = 0
            self.detect(Icamera)
            '''
    def show_index_camera(self):
        for i in range(len(self.cap)):
            self.show_camera_thread(i)

    def release_cameras_clicked(self):
        self.timer_camera.stop()
        len_cap = len(self.cap)
        if len_cap != 0:
            for i in range(len_cap):
                self.cap[i].release()
        self.cap = []

    def Models_Selectionchanged_1(self):	
        # for count in range(self.comboBox_models_1.count()):
        #     print('  ')
        self.model_type = self.comboBox_models.currentText()
        # self.model = self.load_weight()
        print("Model index selection changed ",self.comboBox_models.currentText())             

    def Weights_Selectionchanged_1(self):	
        self.weights = self.comboBox_weights.currentText()
        print("Weight index selection changed ",self.comboBox_weights.currentText()) 
        # f = open("temp/weights/weight_01.txt", "w")
        # f.write(str(self.comboBox_weights.currentText())[8:])
        # print('weight = ',str(self.comboBox_weights.currentText())[8:])
        # f.close()  


    def on_capture(self): 
        for i in range(len(self.cap)):      
            self.logic[i]=1
            
    def detect_index(self):
        for i in range(len(self.cap)):
            self.detect(i)
    
    def detect(self,i):
        global inference
        img_result = [self.imgResult_1]
        gpResult = [self.FinalResult_1]
        # for i in range(len(self.cap)):
        
        if self.logic[i]==1 :
            self.detect_finish[i] = 0
            #cv2.imwrite(f"temp/camera_capture/cam_capture0{i+1}.jpg",self.frame)
            self.time1.setText(time.strftime('%Y/%m/%d-%H:%M:%S'))
            image_dir = f'temp/camera_capture/cam_capture0{i+1}.jpg'
            defect,n_defect,im0 =  yolov5.detect(self.model, img_source=image_dir,img_size=416,conf_thres=0.3,iou_thres=0.4)
            timestr01 = time.strftime('%Y%m%d-%H%M%S')
            if defect > 0 :
                cv2.imwrite('per/images_result/cam_0'+str(i+1)+'/result_cam_capture0'+str(i+1)+'_'+timestr01+'.jpg',im0)
                shutil.copy(image_dir,'per/camera_capture/cam_0'+str(i+1)+'/cam_0'+str(i+1)+'_'+timestr01+'.jpg')
            img_result[i].setPixmap(QtGui.QPixmap(f"temp/images_result/result_cam_0{i+1}.jpg"))
            img_result[i].setScaledContents(True)
            #print('inference = ',inference[i])
            print('defect',defect)
            if defect > 0:
                self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
                self.FinalResult_1.setScaledContents(True)
            else: 
                self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/green.jpg"))
                self.FinalResult_1.setScaledContents(True)

            self.detect_finish[i] = 1
            self.logic[i] = 0
            # self.on_final()           

    def ExitClicked(self):
        print('Exit Button was Clicked')
        self.release_cameras_clicked()
        sys.exit(app.exec_())
        # cap_1.close()
        # cap_2.close()
        window.close()



app = QApplication(sys.argv)
window = camera()
window.show()

try:
    sys.exit(app.exec_())
except:
    print("exiting") 




