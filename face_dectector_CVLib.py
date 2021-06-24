"""     Face Detection using cvlib face detector     """

import cv2
import cvlib as cv
from matplotlib import pyplot as plt

def show_image_with_matplotlib(color_image,title,pos):
    """Shows an image using matplotlib capabilities"""
    img_RGB=color_image[:,:,::-1]
    ax=plt.subplot(1,2,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def detect_face(img,faces):
    """Draws a rectangle over each detected face"""
    for startX,startY,endX,endY in faces:
        cv2.rectangle(img,(startX,startY),(endX,endY),(0,255,0),2)
    return img

img=cv2.imread("face_detection.jpg")
#img1=cv2.resize(img1, (0,0),fx=0.3,fy=0.3)
#img=img1[:,:,::-1]

# Detect faces:
faces,confidences=cv.detect_face(img)

# Draw face detections:
img_1=detect_face(img.copy(), faces)

# Create the dimensions of the figure and set title:
fig=plt.figure(figsize=(10,4))
plt.suptitle("Face Detection using Face_recognition using Cvlib",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_image_with_matplotlib(img_1, "Detector(img,0) :"+str(len(faces)), 1)

# Show the Figure:
plt.show()
