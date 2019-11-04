import glob
import cv2
import pandas as pd
from face_image import All

name = input("Input name : ")
path = 'csv_face/'+name+'_face.csv'

df = pd.read_csv(path)
f_name = []
un_name = []
know_name = []
T_F = []

for file in glob.glob("test_img/"+name+"/*.png"):
    un_name.append(All(file)[0])
    f_name.append(All(file)[1])
    know_name.append(name)
    print(All(file))
    if name == All(file)[0] :
        T_F.append('1')
    else:
        T_F.append('0')

df['F-Name'] = f_name
df['Know-Name'] = know_name
df['Predict-Name'] = un_name
df['T-F'] = T_F
pd.DataFrame.to_csv(df,path)
print("Save CSV to "+path+" Success!")



