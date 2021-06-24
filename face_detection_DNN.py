"""   Face Detection using DNN from Caffe and Tensorflow Face Detector   """

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Load image:
img =cv2.imread("face_detection.jpg")
#print(img.shape[:2])

# Get dimensions of the input image (to be used later):
(h,w)=img.shape[:2]

# Load pre-trained model:
#net=cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000_fp16.caffemodel")
net=cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb","opencv_face_detector.pbtxt")


# Create 4-dimensional blob from image:
blob=cv2.dnn.blobFromImage(img,1.0,(300,300),[104.,117.,123.],False,False)
#print(blob.shape)

# Set the blob as input and obtain the detections:
net.setInput(blob)
detections=net.forward()
#print(detections.shape)
#print(detections[0,0,1,2])

# Initialize the number of detected faces counter detected_faces:
detected_face=0


# Iterate over all detections:
for i in range(0,detections.shape[2]):
    # Get the confidence (probability) of the current detection:
    confidence=detections[0,0,i,2]

    # Only consider detections if confidence is greater than a fixed minimum confidence:
    if confidence > 0.7:
        # Increment the number of detected faces:
        detected_face+=1
        # Get the coordinates of the current detection:
        box=detections[0,0,i,3:7] * np.array([w,h,w,h])
        #print(detections[0,0,i,3:7])
        
        #print(np.array([w,h,w,h]))
        #print(box)
        (startX,startY,endX,endY)=box.astype("int")
        # Draw the detection and the confidence:
        text="{:.3f}%".format(confidence * 100)
        y=startY-10 if startY-10 > 10 else startY+10
        cv2.rectangle(img, (startX,startY), (endX,endY), (255,0,0),3)
        cv2.putText(img, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using OpenCV DNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img, "DNN face detector with TENSORFLOW MODEL: " + str(detected_face), 1)

# Show the Figure:
plt.show()

