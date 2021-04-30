import cv2
import matplotlib.pyplot as plt
import numpy as np

image_name = 'P0000__1__0___0'
image_path = image_name + '.png'
image = cv2.imread(image_path)

mean = 0

var = 1000 # This parameter can be modified to increase/decrease the noise

sigma = var ** 0.5

# Compute the noise
gaussian1 = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
gaussian2 = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
gaussian3 = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))

# Compute the Gaussian noise image
gaussian_image = np.zeros(image.shape, np.float32)
gaussian_image[:, :, 0] = gaussian1
gaussian_image[:, :, 1] = gaussian2
gaussian_image[:, :, 2] = gaussian3
cv2.normalize(gaussian_image, gaussian_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
gaussian_image = gaussian_image.astype(np.uint8)

# Compute the noisy image

noisy_image = np.zeros(image.shape, np.float32)
noisy_image[:, :, 0] = image[:, :, 0] + gaussian1
noisy_image[:, :, 1] = image[:, :, 1] + gaussian2
noisy_image[:, :, 2] = image[:, :, 2] + gaussian3
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
