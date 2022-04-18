
import os, os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

training_images = []
for filename in os.listdir("./training_set"):
    img = cv2.imread(os.path.join("./training_set",filename), cv2.IMREAD_UNCHANGED)
    if img is not None:
        training_images.append(img)


# print(images[0])
# print(images[0].shape)
# print(images[0][100,100])
# plt.imshow(images[0])
# plt.show()


