# Imports
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Image Processing
img = cv2.imread('Other/images/9-15375449.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

# image_np = np.array(img)
# # Sobel Operator: x 
# sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# for i in range(1, image_np.shape[0] - 1):
#     for j in range(1, image_np.shape[1] - 1):
#         img[i, j] = np.sum(sobel_x * image_np[i-1:i+2, j-1:j+2])
# 
# # Sobel Operator: y 
# sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# for i in range(1, image_np.shape[0] - 1):
#     for j in range(1, image_np.shape[1] - 1):
#         sobel_y_current = np.sum(sobel_y * image_np[i-1:i+2, j-1:j+2])
#         img[i, j] = np.sqrt(img[i, j]**2 + sobel_y_current**2)

# Sobel Operator
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude of Sobel Operator
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.show()
