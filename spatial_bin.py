import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutouts/cutout1.jpg')

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def set_color_space(img, color_space):
    if color_space == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if color_space == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color_space == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if color_space == "YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if color_space == "HLS":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
    return img
        
        

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    img = set_color_space(img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features
    
feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()