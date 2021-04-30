import cv2
import matplotlib.pyplot as plt
import numpy as np

image_name = 'P1483__1__0___0'
image_path = image_name + '.png'
image = cv2.imread(image_path)

labels_path = image_name + ".txt"

# Compute bounding boxes

bounding_boxes = []

with open(labels_path, 'r') as f:
    for line in f.readlines():
        line = line.split()
        y_values = [int(float(line[0])), int(float(line[2])), int(float(line[4])), int(float(line[6]))]
        x_values = [int(float(line[1])), int(float(line[3])), int(float(line[5])), int(float(line[7]))]
        bounding_boxes.append((min(x_values), max(x_values), min(y_values), max(y_values)))

mean = 0

var = 1000 # This parameter can be modified to increase/decrease the noise

sigma = var ** 0.5

# Compute the noise
gaussian_image = np.zeros(image.shape, np.float32)

for box in bounding_boxes:
    x_min, x_max, y_min, y_max = box
    gaussian1 = np.random.normal(mean, sigma, (x_max - x_min, y_max - y_min))
    gaussian2 = np.random.normal(mean, sigma, (x_max - x_min, y_max - y_min))
    gaussian3 = np.random.normal(mean, sigma, (x_max - x_min, y_max - y_min))
    gaussian_image[x_min:x_max, y_min:y_max, 0] = gaussian1
    gaussian_image[x_min:x_max, y_min:y_max, 1] = gaussian2
    gaussian_image[x_min:x_max, y_min:y_max, 2] = gaussian3

# Compute the noisy image
noisy_image = image + gaussian_image

# Compute as images
cv2.normalize(gaussian_image, gaussian_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
gaussian_image = gaussian_image.astype(np.uint8)
cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

"""
# Plot images
plt.imshow(image)
plt.imshow(noisy_image)
plt.imshow(gaussian_image)
"""

# Save images
cv2.imwrite(image_name+'_gaussian_noise.png', gaussian_image)
cv2.imwrite(image_name+'_noisy.png', noisy_image)
