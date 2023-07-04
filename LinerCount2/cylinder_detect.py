def detect_cylinder(img_path,cir=0.8,rmin=45,rmax=100) :
    import math
    if ready[0] :
        x1,y1 = int(0.05*frame.shape[1]),int(0.05*frame.shape[0])
        x2,y2 = int(1.0*frame.shape[1]),int(1.0*frame.shape[0])
        min_area,max_area = 10000,30000
    else : 
        x1,y1 = int(0.15*frame.shape[1]),int(0.15*frame.shape[0])
        x2,y2 = int(0.95*frame.shape[1]),int(0.95*frame.shape[0])
        min_area,max_area = 8000,25000
    
    img = frame[y1:y2,x1:x2,:]
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lwhite = np.array([0,0,255-sensitive])
    hwhite = np.array([255,sensitive,255])
    mwhite = cv2.inRange(hsv,lwhite,hwhite)
    #white = cv2.bitwise_and(img,img,mask=mwhite)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
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
        circula = 4*math.PI*area/peri**2
        (xc,yc),r = cv2.minEnclosingCircle(c)
        if circula > cir and (int(r) > rmin and int(r) < rmax) :
        #if cv2.contourArea(c) > min_area and cv2.contourArea(c)< max_area:
            x,y,w,h = cv2.boundingRect(c)
            #xc,yc = x+w/2 ,y+h/2
            ct +=1
            #cv2.circle(img,(int(xc),int(yc)),int(r),(0,255,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,0),2)
            print(area,r,circula)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)
    color = ((0,0,255) if ct != 6 else (0,255,0))
    detect = (ct if detect < ct else detect)
    if not full[0] and not full[1] and not ready[0] and not ready[1] and ct == 6:
        full[0] = True
        
    if full[0] and not full[1] and not ready[0] and not ready[1] and ct != 6 and ct != 0:
        full[0] = False
    if not full[0] or not full[1] and ct == 0 and detect > 0 :
        GPIO.output(12, True) 
    if full[0] or full[1] :
        GPIO.output(12, False)
        
    if full[0] and not full[1] and not ready[0] and not ready[1] and ct == 0:
        ready[0] = True
        
    if ready[0] and full[0] and not full[1] and not ready[1] and ct == 6:
        full[1] = True
    if full[0] and full[1] and ready[0] and not ready[1] and ct != 6 and ct != 0:
        full[1] = False
        
    if full[0] and full[1] and ready[0] and not ready[1] and ct == 0:
        ready[1] = True
        
    if ready[0] and ready[1]:
        GPIO.output(24, False)   
    #reset box           
    if (full[0] and full[1] and ready[0] and ready[1] and ct == 1) or not GPIO.input(16):
        ready[0],ready[1] = False,False
        full[0],full[1] = False,False
        detect = 0
        GPIO.output(24, True)
    
    cv2.putText(img,f'Amount := {ct} pcs',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2)
    cv2.putText(img,f'box1 full {full[0]}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    cv2.putText(img,f'box1 ready {ready[0]}',(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2) 
    cv2.putText(img,f'box2 full {full[1]}',(10,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    cv2.putText(img,f'box2 ready {ready[1]}',(10,190),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    #cv2.imshow('mask',mor_op)
    cv2.imshow('Cont',img)                                                                                           
