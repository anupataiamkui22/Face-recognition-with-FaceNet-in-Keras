import subprocess
from firebase import firebase

url = 'https://facerec-26da7.firebaseio.com/'
messenger = firebase.FirebaseApplication(url)

while 1 :

    result = messenger.get('/device/SwOn', None)
    if result == 1 :
        print('Python RUn !')
        subprocess.check_call("python V4_Voice.py 1", shell=True)
        # subprocess.Popen("python V4_Voice.py 1", shell=True)



