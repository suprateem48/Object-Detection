import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('cutouts/bbox-example-image.jpg')

templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
			'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(255, 255, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for box in bboxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img 


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes 
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    img_copy = np.copy(img)
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    # Read in templates one by one
    # Use cv2.matchTemplate() to search the image
    #     using whichever of the OpenCV search methods you prefer
    # Use cv2.minMaxLoc() to extract the location of the best match
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    method = cv2.TM_CCOEFF_NORMED
    
    for template in template_list:
        template = "cutouts/" + template
        cutout = mpimg.imread(template)
        result = cv2.matchTemplate(img_copy, cutout, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        w,h = cutout.shape[1], cutout.shape[0]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
        
    return bbox_list

bbox_list = find_matches(image, templist)
result = draw_boxes(image, bbox_list)
plt.imshow(result)
plt.show()