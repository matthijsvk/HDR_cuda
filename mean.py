from skimage import io, img_as_float
import numpy as np
import sys

image = io.imread(sys.argv[1])
image = img_as_float(image)
print(np.mean(image))
