# Python program to explain Merging of Channels

# Importing cv2
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Image Editor')

parser.add_argument(
    'imgfile',
    action="store",
    default="img/6.jpg",
    nargs='?')

parser.add_argument(
    'outputFile',
    action="store",
    default='output/imgae_q.jpg',
    nargs='?')

args = parser.parse_args()

# Reading the BGR image using imread() function
image = cv2.imread(args.imgfile)

# Splitting the channels first to generate different
# single

# channels for merging as we don't have separate
# channel images
b, g, r = cv2.split(image)

# Displaying Blue channel image
#cv2.imshow("Model Blue Image", b)

# Displaying Green channel image
#cv2.imshow("Model Green Image", g)

# Displaying Red channel image
#cv2.imshow("Model Red Image", r)

# Using cv2.merge() to merge Red, Green, Blue Channels

# into a coloured/multi-channeled image
image_merge = cv2.merge([r, g, b])

# Displaying Merged RGB image
#cv2.imshow("RGB_Merged_Image", image_merge)

# Use the cvtColor() function to grayscale the image
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', image_gray)

# make sure that you have saved it in the same folder
# You can change the kernel size as you want
blurImg = cv2.blur(image, (10, 10))
#cv2.imshow('blurred image', blurImg)

# Creating the kernel(2d convolution matrix)
kernel1 = np.ones((5, 5), np.float32)/30

# Applying the filter2D() function
img_kernel = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

# Shoeing the original and output image
#cv2.imshow('Kernel Blur', img_kernel)

#Implement this function → bilinear_interpolation
half = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
bigger = cv2.resize(image, (1050, 1610))

stretch_near = cv2.resize(image, (780, 540),
                          interpolation=cv2.INTER_LINEAR)


Titles = ["Original", "Half", "Bigger", "Interpolation Nearest"]
images = [image, half, bigger, stretch_near]
count = 4

"""for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])"""

# Window name in which image is displayed
#window_name = 'bilinear_interpolation'
#bilinear_interpolation image display
#plt.show()

#Implement this function → resize
# Obtaining the Dimensions of the image
height, width = image.shape[:2]

# Downscaling the image
height = int(height/2)
width = int(width/2)

# Performing the resize operation with Nearest neighbor interpolation
image_resize = cv2.resize(image, (width, height),
                          interpolation=cv2.INTER_NEAREST)

# Displaying the image
#cv2.imshow('resize', image_resize)

#90 Degree Rotation
#Implement the function - > rotate
# Window name in which image is displayed
window_name = 'rotate'

# Using cv2.rotate() method
# Using cv2.ROTATE_90_CLOCKWISE rotate
# by 90 degrees clockwise
image_rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Displaying the image
#cv2.imshow(window_name, image_rotate)

#Implement this function → extract_edges
#load the image, convert it to grayscale, and blur it slightly
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# using the Canny edge detector
image_edge = ~cv2.Canny(image_blurred, 10, 200)

#show the output Canny edge maps
#cv2.imshow("Wide Edge Map", image_edge)

#colored_img_quantizer


def quantimage(image, k):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(
        i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


#window_name = 'colored_img_quantizer'
#image = cv2.imread('a.png')
"""plt.imshow(quantimage(image, 5))
plt.show()

plt.imshow(quantimage(image, 8))
plt.show()

plt.imshow(quantimage(image, 25))
plt.show()

plt.imshow(quantimage(image, 35))
plt.show()

plt.imshow(quantimage(image, 45))
plt.show()"""

#Implement this function → mask_generator(image1, image2, mask)
# The kernel to be used for dilation
# purpose
kernel = np.ones((5, 5), np.uint8)

# converting the image to HSV format
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# defining the lower and upper values
# of HSV, this will detect yellow colour
Lower_hsv = np.array([20, 70, 100])
Upper_hsv = np.array([30, 255, 255])

# creating the mask
Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)

# Inverting the mask
mask_yellow = cv2.bitwise_not(Mask)
Mask = cv2.bitwise_and(image, image, mask=mask_yellow)

# Displaying the image
#cv2.imshow('Mask', Mask)

#Cartoonify
# Edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)

# Cartoonization
color = cv2.bilateralFilter(image, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("Image", image)
cv2.imshow("edges", edges)
cv2.imshow("Cartoon", cartoon)


# Saving the image
cv2.imwrite(args.outputFile, cartoon)

print('Successfully saved')

cv2.waitKey(0)

# Window shown waits for any key pressing event
cv2.destroyAllWindows()
