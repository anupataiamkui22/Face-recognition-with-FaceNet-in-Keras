from firebase import firebase
import pyttsx3

url = 'https://facerec-26da7.firebaseio.com/'
messenger = firebase.FirebaseApplication(url)


while 1 :
    result = messenger.get('/device/SwOn', None)
    if result == 1 :
        engine = pyttsx3.init()
        engine.say('Hello I Kuy')
        engine.runAndWait()


