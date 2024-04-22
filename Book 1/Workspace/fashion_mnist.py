import os
import cv2
import urllib
import urllib.request
from zipfile import ZipFile

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}...")
    urllib.request.urlretrieve(URL, FILE)

print("Unzipping images")
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print("Done!")

image_data = cv2.imread('fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)
print(image_data)

import matplotlib.pyplot as plt
image_data = cv2.read('fashion_mnist_images/train/4/0011.ping', cv2.IMREAD_UNCHANGED)
plt.show(image_data)
plt.show()