"""   Face Detection using Dlib Face Detector   """
"""
Face detection using dlib frontal face detector, which is based on Histogram of Oriented Gradients (HOG) features
and a linear classifier in a sliding window detection approach
"""
# Import required packages:
import cv2
import dlib
from matplotlib import pyplot as plt

def detect_face(img,faces):
    """Draws a rectangle over each detected face"""
    for face in faces:
        cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),(0,255,0),10)
    return img

def show_image_with_matplotlib(color_image,title,pos):
    """Shows an image using matplotlib capabilities"""
    img_RGB=color_image[:, :, ::-1]

    ax=plt.subplot(1,2,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Load image and convert to grayscale:
img=cv2.imread("face_detection.jpg")
img_Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detector=dlib.get_frontal_face_detector()


rects_1=detector(img_Gray,0)
rects_2=detector(img_Gray,1)

img_1=detect_face(img.copy(), rects_1)
img_2=detect_face(img.copy(), rects_2)

#print(rects_1)
#print(rects_2)

fig=plt.figure(figsize=(10,4))
plt.suptitle("Face Detection using Dlib Face Detector",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_image_with_matplotlib(img_1, "Detector(ing_Gray,0) :"+str(len(rects_1)), 1)
show_image_with_matplotlib(img_2, "Detector(ing_Gray,1):"+str(len(rects_2)), 2)

plt.show()
