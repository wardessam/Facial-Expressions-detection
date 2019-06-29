import cv2
import pickle
import numpy as np
from tkinter import *

from PIL import Image, ImageTk

root = Tk()
root.geometry('700x500')
root.title("Object Tracking")

root.configure(width=800, height=400, background="#2f4f4f")

lbl = Label(root, text="Welcome To Our Program!", font=("Helvetica", 23), bg="#2f4f4f", fg="#8B8B7A")
lbl.pack()
lbl.place(x=160, y=15)
img = ImageTk.PhotoImage(Image.open("image.jpg"))
lbl2 = Label(root, image=img)
lbl2.pack()
lbl2.place(x=70, y=150)


def track_obj():
    root.destroy()
    FaceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    Recognizer = cv2.face.LBPHFaceRecognizer_create()
    # rec is used in training and predicting the face

    Recognizer.read("train.yml")
    # reads the yml table made in (Iqmages To Train .py)

    oldLabels = {"Expression_name": 1}

    with open("labels.pickle", 'rb')as f:
        openLabels = pickle.load(f)
        # opens the pickle file as rb =Read binary and load the file
        Labels = {a: b for b, a in openLabels.items()}
       # takes dictionary comprises labels items key and values


    videoCap = cv2.VideoCapture(0)
      # opening the camera (0 is for your main camera) (capdshow DirectShow (via videoInput))

    while True:
        _, frame = videoCap.read()
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # converting the picture to grey because this is the only way to compare (cant compare with colors)
        faces = FaceCascade.detectMultiScale(grayImage, scaleFactor=1.5, minNeighbors=5)
        # detects objects of different sizes in the input image (gray)

        for (x, y, w, h) in faces:

         roi_gray = grayImage[y:y + h, x:x + w]
         roi_color = frame[y:y + h, x:x + w]
         # these to make the width and hight as in the picture of the face

         iD, conf = Recognizer.predict(roi_gray)

         # predict from the training model rec
         # like angry happy sad

         if conf >= 45:
            print(iD)
            print(Labels[iD])

            font = cv2.FONT_ITALIC
            name = Labels[iD]
            cv2.putText(frame, name, (x, y - 4), font, 1, (255, 0, 0), 1)

            # a design for the lable name that is going to appear in the program

        # DRAW RECTANGLE
         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

  #cv2.imwrite('me.jpg', frame)
         cv2.imshow('Face Detection', frame)
         key = cv2.waitKey(1)
         if key == ord('q'):
          break
    videoCap.release()
    cv2.destroyAllWindows()

btn = Button(root, text="Start?", bg="#FFB90F", fg="#8B8B7A", command=track_obj)
btn.config(width=30, height=1)
btn.pack()
btn.place(x=230, y=90)
# this makes the window in infinite loop until i close it, without this line the window will appear and disappear in a second
root.mainloop()