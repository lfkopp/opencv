import cv2
import numpy as np
import os
#print(cv2.__version__)
# pip install opencv_python-3*win_amd64.whl

path1 = "opencv/photos/"
face_cascade = cv2.CascadeClassifier('opencv/xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/xml/haarcascade_eye.xml')

count = 1
listing = os.listdir(path1)
for file in listing:
    print(str(file))
    img = cv2.imread(path1 + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2,2)
    for (x,y,w,h) in faces:
        crop_im  = img[y: y + h, x: x + w]
        crop_im = cv2.resize(crop_im, (100,100))
        cv2.imwrite('opencv/caras/cara'+str(count)+'.jpg',crop_im)
        count = int(count) + 1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
    cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
