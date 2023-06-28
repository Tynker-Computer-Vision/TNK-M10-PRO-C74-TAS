import numpy as np
import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector


path = "Face-image-dataset-small"
images = []
ages = []

detector = FaceDetector()

for img in os.listdir(path):
    try:
        print(img)
        if img!='.git':
            age = img.split("_")[0]
            genders = img.split("_")[1]
            img = cv2.imread(str(path)+"/"+str(img))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            # Detect the face in the image and store the result in img, bbox variables
            img, bbox = detector.findFaces(img, draw=False) 
 
            # if bbox exits then
            if bbox:
                # Get the X,Y,W,H of bbox in respective variables i.e X, Y, W, H
                X, Y, W, H = bbox[0]['bbox']

                # Crop the face out of the image
                croppedImg = img[Y:Y+H, X:X+W]
                # Resize the image to 200, 200
                resizedImg = [cv2.resize(croppedImg, (200, 200))]
                        
                # Append resized image to images array        
                images.append(resizedImg)
                
                ages.append(age)
    except:
        print("error in reading")

ages = np.array(ages,dtype=np.int64)

# Convert images to np.array
images = np.array(images)

print("Age",ages)

# Print images array
print(images)
