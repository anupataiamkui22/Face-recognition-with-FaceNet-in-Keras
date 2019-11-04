import pyrebase

config = {
  "apiKey": "AIzaSyCSEk3ZPpeh5cV8mlODks2CQh80pRUp5oY",
  "authDomain": "facerec-26da7.firebaseapp.com",
  "databaseURL": "https://facerec-26da7.firebaseio.com",
  "storageBucket": "facerec-26da7.appspot.com"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()

def stream_handler(message):
    print(message["event"]) # put
    print(message["path"]) # /air
    print(message["data"]) # ON or OFF

my_stream = db.stream(stream_handler)