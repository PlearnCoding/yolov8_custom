import cv2

def roundCorner(x, img, color=(0,255,0), factor=0.10, thickness=2):
    x1,y1 = int(x[0]), int(x[1])
    x2,y2 = int(x[2]), int(x[3])
    l = int(min(x2-x1,y2-y1)*factor)
    cv2.ellipse(img,(x1+l,y1+l),(l,l),0,-180,-90,color,thickness,0)
    cv2.ellipse(img,(x2-l,y2-l),(l,l),0,0,90,color,thickness,0)
    cv2.ellipse(img,(x2-l,y1+l),(l,l),0,-90,0,color,thickness,0)
    cv2.ellipse(img,(x1+l,y2-l),(l,l),0,-180,-270,color,thickness,0)
    cv2.line(img, (x1+l,y1), (x2-l,y1), color, thickness,0)
    cv2.line(img, (x1+l,y2), (x2-l,y2), color, thickness,0)
    cv2.line(img, (x1,y1+l), (x1,y2-l), color, thickness,0)
    cv2.line(img, (x2,y1+l), (x2,y2-l), color, thickness,0)

img = cv2.imread('test.png')
roundCorner([100,100,400,400],img)
cv2.imshow('test',img)
cv2.waitKey()
cv2.destroyAllWindows()