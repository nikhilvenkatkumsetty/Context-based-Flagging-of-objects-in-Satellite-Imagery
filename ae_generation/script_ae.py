import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch

# TO DO: DEFINE THE PATH TO THE MODEL
MODEL_PATH = "" 
NUMBER_OF_CLASSES = 4

# Load the model
model = torch.load(MODEL_PATH)

# TO DO: MEAN / STD SHOULD BE MODIFIED IF NECESSARY
preprocessing = (0, 1)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=NUMBER_OF_CLASSES, preprocessing=preprocessing)

# Define images and labels
# TO DO: images, labels SHOULD BE DEFINED
images, labels = "images", "labels"
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))

attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))


# Show the first adversarial image
image = images[0]
adversarial = attack(images[:1], labels[:1])[0]

# CHW to HWC
image = image.transpose(1, 2, 0)
adversarial = adversarial.transpose(1, 2, 0)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()



