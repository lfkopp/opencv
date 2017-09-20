
import cv2
#print(cv2.__version__)

cascade_src = 'opencv/cars.xml'
video_src = 'opencv/dados/video01.mp4'
#video_src = 'opencv/dados/video02.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

count = 0
frame = 1
while True:
    cap.set(1, 80 * frame)
    frame = frame + 1
    print(frame)
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 1)

    for (x,y,w,h) in cars:
        if w > 100:
            crop_im  = img[y: y + h, x: x + w]
            cv2.imwrite('opencv/dados/cuts/carro'+str(count)+'.jpg',crop_im)
            count = int(count) + 1
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            text_color = (255,0,0)
            cv2.putText(img, str(count), (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1)

    cv2.imshow('video', img)


    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
