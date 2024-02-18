import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = '../../PycharmProjects/TeamAnantPayload/clocktower.jpg'

image = cv.imread(path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
flattened_image = image_rgb.reshape((-1, 3))  # Flatten the image to a 1D array

height, width = image.shape[:2]

matrix = np.array(image_rgb)

for j in range (0,width-1,2):
    for i in range (0,height-1,2):
        arr1 = matrix[i,j]
        arr2 = matrix[i,j+1]
        arr3 = matrix[i+1,j]
        arr4 = matrix[i+1,j+1]
        arr = arr1//4 + arr2//4 + arr3//4 + arr4//4
        matrix[i,j] = matrix[i+1,j] = matrix[i,j+1] = matrix[i+1,j+1] = arr


plt.imshow(matrix)
plt.show()


