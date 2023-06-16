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
            img = cv2.imread(str(path)+"/"+str(img))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            # Detect the face in the image and store the result in img, bbox variables
            img, bbox = detector.findFaces(img, draw=False) 
 
            # if bbox exits then
            if bbox:
                # Get the X,Y,W,H of bbox in respective variables i.e X, Y, W, H
                X = bbox[0]['bbox'][0]
                Y = bbox[0]['bbox'][1]
                W = bbox[0]['bbox'][2]
                H = bbox[0]['bbox'][3]

                # Crop the face out of the image
                croppedImg = img[Y:Y+H, X:X+W]
                # Resize the image to 200, 200
                resizedImg = [cv2.resize(croppedImg, (200, 200))]
                        
                # Append resized image to images array        
                images.append(resizedImg)
                
                ages.append(age)
    except:
        print("error in reading")

# Convert ages to np.array
ages = np.array(ages,dtype=np.int64)

# Convert images to np.array
images = np.array(images)

# Print age array
print("Age",ages)

# Print images array
print(images)
