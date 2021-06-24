"""    Face Detection using HAAR feature-based cascade classifiers    """

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load cascade classifiers:
cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
cascade1=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load image and convert to grayscale:
img=cv2.imread("face_detection.jpg")
imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces:
faces_alt2=cascade.detectMultiScale(imgGray)
faces_default=cascade1.detectMultiScale(imgGray)

retval,face_haar_alt2=cv2.face.getFacesHAAR(img,"haarcascade_frontalface_alt2.xml")
face_haar_alt2=np.squeeze(face_haar_alt2)

retval,face_haar_default=cv2.face.getFacesHAAR(img,"haarcascade_frontalface_default.xml")
face_haar_default=np.squeeze(face_haar_default)

#cv2.imshow("Output", img)
#cv2.waitKey(0)
def detect_face(img,faces):
    """Draws a rectangle over each detected face"""
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)
    return img

def show_image_with_matplotlib(color_image,title,pos):
    """Shows an image using matplotlib capabilities"""    
    img_RGB=color_image[:,:,::-1]

    ax=plt.subplot(2,2,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Draw face detections:
img_faces_alt2=detect_face(img.copy(),faces_alt2)
img_faces_default=detect_face(img.copy(),faces_default)

img_face_haar_alt2=detect_face(img.copy(),face_haar_alt2)
face_haar_default=detect_face(img.copy(),face_haar_default)

# Create the dimensions of the figure and set title:
fig=plt.figure(figsize=(10,8))
plt.suptitle("Face detection using haar feature-based cascade classifiers",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_image_with_matplotlib(img_faces_alt2,"detectMultiScale(frontalface_alt2): "+str(len(faces_alt2)),1)
show_image_with_matplotlib(img_faces_default,"detectMultiScale(frontalface_default):"+str(len(faces_default)),2)
show_image_with_matplotlib(img_face_haar_alt2,"getFacesHAAR(frontalface_alt2): "+str(len(face_haar_alt2)),3)
show_image_with_matplotlib(face_haar_default,"getFacesHAAR(frontalface_default): "+str(len(face_haar_default)),4)

# Show the Figure:
plt.show()
