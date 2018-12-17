import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

def loadImage(image_name):
    """
    image_name: the full path name of the image (including the image's name).
    Returns a flattened image of shape (1, Width * Heigh)
    """
    from scipy import misc
    image = misc.imread(image_name)
    print("Reading image:", image_name)
    # return image.reshape((1,-1))
    return image

path = "training/positive/C1R1E0N1.TIF"
image = loadImage(path)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()