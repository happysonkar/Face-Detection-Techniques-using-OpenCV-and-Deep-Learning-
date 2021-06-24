"""   Face Detection unsing Dlib CNN Face Detector   """
import cv2
import dlib
from matplotlib import pyplot as plt

def detect_face(img,faces):
    for face in faces:
        """Draws a rectangle over each detected face"""
        # faces contains a list of mmod_rectangle objects
        # The mmod_rectangle object has two member variables, a dlib.rectangle object,and a confidence score
        # Therefore, we iterate over the detected mmod_rectangle objects accessingdlib.rect to draw the rectangle
        cv2.rectangle(img,(face.rect.left(),face.rect.top()),(face.rect.right(),face.rect.bottom()),(0,255,0),3)
    return img

def show_image_with_matplotlib(color_image,title,pos):
    img_RGB=color_image[:, :, ::-1]

    ax=plt.subplot(1,1,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Load image and convert to grayscale:
img=cv2.imread("face_detection.jpg")
img_Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Load CNN detector from dlib:
cnn_face_detector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Detect faces:
rects_1=cnn_face_detector(img,0)
#rects_2=cnn_face_detector(img,1)

# Draw face detections:
img_1=detect_face(img.copy(), rects_1)


# Create the dimensions of the figure and set title:
fig=plt.figure(figsize=(10,4))
plt.suptitle("Face Detection unsing Dlib CNN Face Detector",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_image_with_matplotlib(img_1, "Detector(ing_Gray,0) :"+str(len(rects_1)), 1)
#show_image_with_matplotlib(img_2, "Detector(ing_Gray,1):"+str(len(rects_2)), 2)

# Show the Figure:
plt.show()
