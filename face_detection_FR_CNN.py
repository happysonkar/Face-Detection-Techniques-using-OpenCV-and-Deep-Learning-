"""  Face Detection using Face_recognition using CNN Model  """
import cv2
import dlib 
import face_recognition
from matplotlib import pyplot as plt

def show_image_with_matplotlib(color_image,title,pos):
    """Draws a rectangle over each detected face"""
    img_RGB=color_image[:,:,::-1]
    ax=plt.subplot(1,2,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def detect_face(img,faces):
    for top,right,bottom,left in faces:
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
    return img

# Load image:
img1=cv2.imread("face_detection.jpg")
img1=cv2.resize(img1, (0,0),fx=0.3,fy=0.3)
img=img1[:,:,::-1]
#img_Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Perform face detection using face_recognition (internally calls dlib HOG face detector):
rects_1=face_recognition.face_locations(img,0,"cnn")
rects_2=face_recognition.face_locations(img,1,"cnn")

# Draw face detections:
img_1=detect_face(img1.copy(), rects_1)
img_2=detect_face(img1.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig=plt.figure(figsize=(10,4))
plt.suptitle("Face Detection using Face_recognition using CNN",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_image_with_matplotlib(img_1, "Detector(img,0) :"+str(len(rects_1)), 1)
show_image_with_matplotlib(img_2, "Detector(img,1):"+str(len(rects_2)), 2)

# Show the Figure:
plt.show()
