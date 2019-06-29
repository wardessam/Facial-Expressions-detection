import os
from PIL import Image
import numpy as np
import cv2
import pickle

FaceCascade = cv2.CascadeClassifier('C:\\Users\\Yousef\\PycharmProjects\\StatProject\\haarcascade_frontalface_default.xml')
Recognizer = cv2.face.LBPHFaceRecognizer_create()
#the same as in OpenCamera
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#removes the last segment of a path.
#__file__  points to the filename of the current module
image_dir =os.path.join(BASE_DIR,"Images")
#it returns the images paths in each folder of" Images"

ExpCount = 0
Label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png")or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label =os.path.basename(root)
            #it returns the folder name (basename) in lower alphabetics
            print (path,label)
            #just to check if it is working
            if not label in Label_ids:
                Label_ids[label] = ExpCount
                ExpCount += 1
                #giving the images ids

            id = Label_ids[label]

            pil_image=Image.open(path).convert("L")
            #it opens the image and convert it into 'L' (gray)

            size=(550,550)

            final_image=pil_image.resize(size,Image.ANTIALIAS)
            #resize all images and faces the same size (500,500)

            image_array=np.array(final_image,"uint8")
            faces =FaceCascade.detectMultiScale(image_array, scaleFactor=1.5 ,  minNeighbors=5)
            for(x,y,w,h)in faces:
                roi =image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id)
            # same as in OpenCamera


with open("labels.pickle",'wb')as f:
    pickle.dump(Label_ids,f)
    #this takes the id and the picture and wirte them in a pickle file
    # wb=write binary in the pickle file


Recognizer.train(x_train,np.array(y_labels))
Recognizer.save("train.yml")
#and here we train the program with the folders and
#subfolders and puting all the pics in the YAML (yml) file
