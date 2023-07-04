import imutils
import cv2 ,time ,datetime
from threading import Thread

class Camera:
    def __init__(self):
        # super().__init__()
        self.frame = None
        self.video = cv2.VideoCapture(0)
        self.f_width = 1280
        self.f_height = 720
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH,self.f_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT,self.f_height)
        self._start = True
        self.th = Thread(target=self.getFrame)
        self.rotation = 0

    def getFrame(self):
        while self._start :
            (ret,_frame) = self.video.read()
            _frame = imutils.rotate_bound(_frame,self.rotation)
            # self.frame = cv2.cvtColor(_frame,cv2.COLOR_BGR2GRAY)
            self.frame = _frame
            cv2.imshow('test',self.frame)
            k = cv2.waitKey(5)
            if k == ord('s') :
                cv2.imwrite(time.strftime('%Y%m%d-%H%M%S')+'.jpg',self.frame)
            elif k == ord('q') : 
                self.stop()
                
        cv2.destroyAllWindows()
        self.video.release()

    def start(self):
        self._start = True
        self.th.start()
        
    def stop(self):
        self._start = False
        #self.th.stop()
        
    
camera1 = Camera()
camera1.start()

#a = input('Enter your name :' )
#while a != 'y' :
    #print('your name is :',a)
    #a = input('Enter your name :' )
# cv2.imshow('test',camera1.frame)


