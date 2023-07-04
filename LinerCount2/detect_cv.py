import sys,os ,shutil
import cv2
import numpy as np
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
# import RPi.GPIO as GPIO
#import pyautogui
from linenotify.lineNotify import LineNotify
# im1 = pyautogui.screenshot()
# im = pyautogui.screenshot(region=(0,0, 300, 400))
line = LineNotify()
timestr01 = time.strftime('%Y%m%d-%H%M%S')
inference = [0]

class camera(QMainWindow): #เป็น class ลูกของ QMainWindow
    def __init__(self):
        super(camera, self).__init__()
        loadUi('qtdesigner_detect_1r1_hd.ui', self)  
        # model_efficientdet_keras
        self.relay_pin = 24
        self.sensor_pin = 25
        #self.alarm_pin = 12
        # GPIO.setwarnings(False) 
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self.relay_pin, GPIO.OUT)
        # GPIO.setup(self.sensor_pin, GPIO.IN)
        # GPIO.output(self.relay_pin, False)
        # GPIO.setup(self.alarm_pin, GPIO.IN,pull_up_down=GPIO.PUD_UP)
        self.camera_no = 'test_CL.avi'
        self.sensitive = 70
        self.detec_time = 1.0
        self.level_factor = 0.17
        self.kernel = 5
        self.debug = 'Run'
        self.circ, self.rmin, self.rmax = 0.75,100,200

        self.full1,self.full2 = False,False
        self.ready1,self.ready2 = False,False
        self.count1 = 0
        self.count2 = 0
        self.count_box = 0
        self.model_name = 'RT155'
        self.detect = 0
        self.frame = None
        self.cap = []
        self.N = 1   
        self.defect = [0]*self.N
        self.logic = [0]*self.N
        self.value = [0]*self.N
        self.detect_finish = [0]*self.N
        self.recordT = time.time()
        #self.th = Thread(target=self.detect,args=[0])

        self.lineCamera.setText(str(self.camera_no))
        self.lineLight.setText(str(self.sensitive))
        self.lineTime.setText(str(self.detec_time))
        self.lineFactor.setText(str(self.level_factor))
        self.lineKernel.setText(str(self.kernel))
        self.lineCirc.setText(str(self.circ))
        self.lineRmin.setText(str(self.rmin))
        self.lineRmax.setText(str(self.rmax))
        self.lineDebug.setText(str(self.debug))

        self.cmdConnect.clicked.connect(self.open_cameras_clicked) 
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.setInterval(33)
        self.timer_camera.timeout.connect(self.show_index_camera)
        part = ['RT155','RT140','RT110']
        
        self.cmdExit.clicked.connect(self.ExitClicked)
        self.actionExit.triggered.connect(self.ExitClicked)
        self.actionWeights.triggered.connect(self.ExitClicked)

        self.comboBox_Part.setItemText(0,part[0])
        self.comboBox_Part.setItemText(1,part[1])
        self.comboBox_Part.setItemText(2,part[2])

        self.cmdReset.clicked.connect(self.reset)        

    def on_final(self):        
        # print(self.detect_finish_1, self.detect_finish_2, self.detect_finish_3, self.detect_finish_4)
        global inference
        # if self.detect_finish_1==1 or self.detect_finish_2==1 or self.detect_finish_3==1 or self.detect_finish_4==1:
        pass
        # if self.detect > 0:
        #     self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/green.jpg"))
        #     self.FinalResult_1.setScaledContents(True)                
        # else:
        #     self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
        #     self.FinalResult_1.setScaledContents(True) 

    def load_init(self):
        # GPIO.output(self.relay_pin, False)
        if self.lineCamera.text().isdigit():
            self.camera_no = int(self.lineCamera.text())
        else :
            self.camera_no = str(self.lineCamera.text())
        self.sensitive = int(self.lineLight.text())
        self.detec_time = float(self.lineTime.text())
        self.level_factor = float(self.lineFactor.text())
        self.kernel = int(self.lineKernel.text())
        self.circ = float(self.lineCirc.text())
        self.rmin = int(self.lineRmin.text())
        self.rmax = int(self.lineRmax.text())
        self.debug = True if self.lineDebug.text() == 'debug' else False
        self.model_name = self.comboBox_Part.currentText()
        
    def reset(self):
        self.release_cameras_clicked()
        self.camera_no = 0
        self.sensitive = 200
        self.detec_time = 3
        self.level_factor = 0.1
        self.kernel = 5
        self.full1,self.full2 = False,False
        self.ready1,self.read2 = False,False
        self.count1 = 0
        self.count2 = 0
        self.camera_display1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
        self.camera_display1.setScaledContents(True)
        self.imgResult1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
        self.imgResult1.setScaledContents(True)

    def reset_cycle(self):
        self.full1,self.full2 = False,False
        self.ready1,self.read2 = False,False
        self.count1,self.count2 = 0,0
        self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
        self.FinalResult_1.setScaledContents(True)
        self.lblFull1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
        self.lblFull1.setScaledContents(True)
        self.lblBox1.setText(str(self.count1))
        self.lblFull2.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
        self.lblFull2.setScaledContents(True)
        self.lblBox2.setText(str(self.count2))

    def open_cameras_clicked(self):
        self.load_init()
        for i in range(self.N):
            # temp = cv2.VideoCapture(i,cv2.CAP_DSHOW)
            temp = cv2.VideoCapture(self.camera_no)
            # temp = cv2.VideoCapture('cl-count1.avi')
            temp.set(3,640); temp.set(4,480)
            if temp.isOpened() == True:
                self.cap.append(temp)
            else:
                temp.release()
                print('error open camera')
        print('click open camera')
        self.timer_camera.start()
        self.recordT = time.time()

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
            self.release_cameras_clicked()
        else:
            self.displayImage(displayI[Icamera],self.frame,1)
            # cv2.waitKey()
            tic = time.time()
            if (tic - self.recordT) >= self.detec_time:
                self.logic[Icamera]==1
                cv2.imwrite(f"temp/camera_capture/cam_capture0{Icamera+1}.jpg",self.frame)
                self.recordT = time.time()
                self.detection(Icamera)

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

    def on_capture(self): 
        for i in range(len(self.cap)):      
            self.logic[i]=1 

    def cornerRectangle(self,x, img, colorB=(0,255,255),colorR=(0,255,0), factor=0.12, thicknessB=1,thicknessR=2):
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

    def roundCorner(self,x, img, colorB=(0,255,0),colorR=(0,255,0), factor=0.12, thicknessB=1,thicknessR=2):
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

    def detect_cylinder(self,img_path,cir=0.8,dmin=45,dmax=100,debug=False):
        import math
        image = cv2.imread(img_path)
        if not self.ready1 :
            x1,y1 = int(self.level_factor*1.35*image.shape[1]),int(self.level_factor*1.35*image.shape[0])
            x2,y2 = int((1-self.level_factor*1.35)*image.shape[1]),int((1-self.level_factor*1.35)*image.shape[0])
            # min_area,max_area = 10000,30000
        else : 
            x1,y1 = int(self.level_factor*image.shape[1]),int(self.level_factor*image.shape[0])
            x2,y2 = int((1-self.level_factor)*image.shape[1]),int((1-self.level_factor)*image.shape[0])
            # min_area,max_area = 8000,25000
        
        img = image[y1:y2,x1:x2,:]
        img = cv2.resize(img,(image.shape[1],image.shape[0]))
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        lwhite = np.array([0,0,255-self.sensitive])
        hwhite = np.array([255,self.sensitive,255])
        mwhite = cv2.inRange(hsv,lwhite,hwhite)
        #white = cv2.bitwise_and(img,img,mask=mwhite)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kernel,self.kernel))
        #mor_cl = cv2.morphologyEx(mwhite,cv2.MORPH_CLOSE,kernel)
        mor_op = cv2.morphologyEx(mwhite,cv2.MORPH_OPEN,kernel)
        mor_oper = cv2.erode(mor_op,None,iterations=1)

        cnts,thre = cv2.findContours(mor_op,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = (15 if len(cnts)>15 else len(cnts))
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:cnt]
        ct = 0
        
        for c in cnts:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c,True)
            circula = 4*math.pi*area/peri**2
            # (xc,yc),r = cv2.minEnclosingCircle(c)
            x,y,w,h = cv2.boundingRect(c)
            d = min(w,h)
            if circula > cir and (int(d) > dmin and int(d) < dmax) :
            #if cv2.contourArea(c) > min_area and cv2.contourArea(c)< max_area:
                # x,y,w,h = cv2.boundingRect(c)
                #xc,yc = x+w/2 ,y+h/2
                ct +=1
                #cv2.circle(img,(int(xc),int(yc)),int(r),(0,255,255),2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),1)
                # self.cornerRectangle(img,x,y,x+w,y+h,color=(0,255,0),factor=0.17, thickness=2)
                # self.cornerRectangle((x,y,x+w,y+h), img, colorB=(0,255,255),colorR=(0,255,0), factor=0.12, thicknessB=2,thicknessR=3)
                self.roundCorner((x,y,x+w,y+h), img, colorB=(0,255,255),colorR=(0,255,0), factor=0.12, thicknessB=2,thicknessR=3)
                print(f'Dia : {d} pixel, Cirula : {circula:0.2f} , Area : {int(area)} pixel2')
        self.detect = ct
        print('Detect :',self.detect)
        if debug:
            cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)

        self.lblCount.setText(str(self.count_box))
        if not self.full1 :
            if self.count1 <= self.detect:
                self.count1 = self.detect
                self.lblFull1.setPixmap(QtGui.QPixmap("temp/info/yellow.png"))
                self.lblFull1.setScaledContents(True)
                self.lblBox1.setText(str(self.count1))
                self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
                self.FinalResult_1.setScaledContents(True)
                if self.count1 == 6:
                    self.full1 = True
                    #im1 = pyautogui.screenshot('my_screenshot1.png')
                    #line.lineNotifyFile('my_screenshot1.png')
            else :
                if self.detect == 0:
                    self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
                    self.FinalResult_1.setScaledContents(True)
                    print('Not full in Box#1 ???')
        else :
            self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/gray.png"))
            self.FinalResult_1.setScaledContents(True)
            self.lblFull1.setPixmap(QtGui.QPixmap("temp/info/green.png"))
            self.lblFull1.setScaledContents(True)
            self.lblBox1.setText(str(self.count1))
            if self.detect == 0:
                self.ready1 = True
            else :
                if not self.ready1:
                    print('Please put cover#1 !!!!')
            if self.ready1 :
                if not self.full2 :
                    if self.count2 <= self.detect:
                        self.count2 = self.detect
                        self.lblFull2.setPixmap(QtGui.QPixmap("temp/info/yellow.png"))
                        self.lblFull2.setScaledContents(True)
                        self.lblBox2.setText(str(self.count2))
                        if self.count2 == 6:
                            self.full2 = True
                            #im2 = pyautogui.screenshot('my_screenshot2.png')
                            #line.lineNotifyFile('my_screenshot2.png')
                    else :
                        if self.detect == 0:
                            self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
                            self.FinalResult_1.setScaledContents(True)
                            print('Not full in Box#2 ???')
                else :
                    self.lblFull2.setPixmap(QtGui.QPixmap("temp/info/green.png"))
                    self.lblFull2.setScaledContents(True)
                    self.lblBox2.setText(str(self.count2))
                    if self.detect == 0:
                        self.ready2 = True
                    else :
                        if not self.ready2:
                            print('Please put cover#2 !!!!') 
                    if self.ready2 :#and GPIO.input(self.sensor_pin)==1:
                        # GPIO.output(self.relay_pin, True)
                        print('All Ready to Push Box')
                        self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/green.png"))
                        self.FinalResult_1.setScaledContents(True)
                    if self.ready2 and (self.detect == 0) :# and GPIO.input(self.sensor_pin)==0:
                        self.count_box += 1
                        print('Reset new cycle')
                        self.reset_cycle()
                        time.sleep(1)
                        # GPIO.output(self.relay_pin, False)
                        
        return self.detect,img   
    
    def detection(self,i):
        global inference
        img_result = [self.imgResult_1]
        gpResult = [self.FinalResult_1]
        self.detect_finish[i] = 0
        #cv2.imwrite(f"temp/camera_capture/cam_capture0{i+1}.jpg",self.frame)
        self.time1.setText(time.strftime('%Y/%m/%d-%H:%M:%S'))
        image_dir = f'temp/camera_capture/cam_capture0{i+1}.jpg'

        n_defect,im0 =  self.detect_cylinder(image_dir,self.circ,self.rmin,self.rmax,self.debug)
        timestr01 = time.strftime('%Y%m%d-%H%M%S')
        cv2.imwrite('temp/images_result/result_cam_0'+str(i+1)+'.jpg',im0)
        # if n_defect > 0 :
        #     cv2.imwrite('per/images_result/cam_0'+str(i+1)+'/result_cam_capture0'+str(i+1)+'_'+timestr01+'.jpg',im0)
        #     shutil.copy(image_dir,'per/camera_capture/cam_0'+str(i+1)+'/cam_0'+str(i+1)+'_'+timestr01+'.jpg')
        img_result[i].setPixmap(QtGui.QPixmap(f"temp/images_result/result_cam_0{i+1}.jpg"))
        img_result[i].setScaledContents(True)
        #print('inference = ',inference[i])
        # print('detect',n_defect)
        # if n_defect > 0:
        #     self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/red.jpg"))
        #     self.FinalResult_1.setScaledContents(True)
        # else: 
        #     self.FinalResult_1.setPixmap(QtGui.QPixmap("temp/info/green.jpg"))
        #     self.FinalResult_1.setScaledContents(True)

        self.detect_finish[i] = 1
        self.logic[i] = 0
        self.on_final()           

    def ExitClicked(self):
        print('Exit Button was Clicked')
        self.release_cameras_clicked()
        window.close()
        sys.exit(app.exec_())
        # cap_1.close()
        # cap_2.close()
        

app = QApplication(sys.argv)
window = camera()
window.show()

try:
    sys.exit(app.exec_())
except:
    print("exiting") 




