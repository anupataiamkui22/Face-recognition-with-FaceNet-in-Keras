import cv2
from firebase import firebase
from test_camera import save_img

# cap = cv2.VideoCapture("http://192.168.43.3:53924/videostream.cgi?user=admin&pwd=88888888")
cap = cv2.VideoCapture("rtsp://admin:88888888@192.168.43.3:10554/tcp/av0_0")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
url = 'https://facerec-26da7.firebaseio.com/'
messenger = firebase.FirebaseApplication(url)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()