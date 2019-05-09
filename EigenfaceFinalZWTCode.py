
#%%
# Import necessary packages
from __future__ import print_function
import pathlib
from pathlib import Path
import os
import sys
import cv2
import numpy as np
from PIL import Image
import csv
import shutil
import random
import glob


#%%
#make file alias to make my life easier
path = Path('path commented out for privacy')


#%%
list(path.glob('*.jpg'))


#%%
print(path)


#%%
##############################################################################
#Create data matrix from a list of images
#Allocate space for all images in one data matrix.
#w = width of an image in the dataset.
#h = height of an image in the dataset.
#3 is for the 3 color channels.
############################################################################

def createDataMatrix(path):
    print("Creating data matrix",end=" ... ") 
#This function counts the number of pictures in the directory
    numImages = len(path)
    sz = path[0].shape
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i,:] = image
    
    print("DONE")
    return data


#%%
#Eigen step 1 -Run this piece of code first in the series after testing your dependencies and files
# Read images from the directory 
def readImages(path):
    print("Reading images from " + path, end="...")

    # Create array of array of images.  
    images = []
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        print(fileExt)
        print(filePath)
        if fileExt in [".jpg", ".jpeg"]:

            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath) #This is how OpenCV opens image files
            print(im)

            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:

                # Convert image to floating point
                im = np.float32(im)/255.0
                images.append(im) # Add image to list
                imFlip = cv2.flip(im, 1);# Flip image 
                images.append(imFlip)# Append flipped image to dictionary

    numImages = int(len(images) / 2)# Exit if no image found
    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images


#%%
#Eigen step 2
#this is how the interpreter notes that this is to be run first is the main code
if __name__ == '__main__': 

    # Number of EigenFaces
    NUM_EIGEN_FACES = 10

    # Maximum weight
    MAX_SLIDER_VALUE = 255

    # Directory containing images
    dirName = ''
 
    # Read images
    images = readImages(dirName)
    #images = cv2.imread(imgdr)
    
    # Size of images
    sz = np.size(images)
    #sz = np.array(images)
    #sz = images[0].shape   
    
    # Create data matrix for PCA.
    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print ("DONE")

    averageFace = mean.reshape(sz)

    eigenFaces = []; 

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    # Create window for displaying Mean Face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
    
    # Display result at 2x size
    output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []
    
    # Create Trackbars
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE/2))
        cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
    
    # You can reset the sliders by clicking on the mean image.
    cv2.setMouseCallback("Result", resetSliderValues);
    
    print('''Usage: Change the weights using the sliders, Click on the result window to reset sliders, Hit ESC to terminate program.''')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%% [markdown]
# The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. ... For example : we want to make a multiply function that takes any number of arguments and able to multiply them all together. It can be done using *args.

#%%
# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
    # Start with the mean image
    output = averageFace
    
    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        #OpenCV does not allow slider values to be negative. 
        #So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
         
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
    createNewFace()


#%%



