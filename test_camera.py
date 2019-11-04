import numpy as np
import cv2

def save_img(cap,face_cascade,i) :
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            roi = (w*h)/100
            if (roi >= 85   ) :
                cv2.imwrite("test_img/test.jpg",img)
                return gray
                exit()
        # cv2.waitKey(100)
        # cv2.imshow('123',img)
    # cap.release()
    # cv2.destroyAllWindows()
