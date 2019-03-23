import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import data
from skimage.feature import match_template

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# normal_img_path = "./dataset/training/s1/normal.tif"
# defect_img_path = "./dataset/training/s1/defect.tif"

normal_img_path = "./dataset/training/s3/normal.tif"
defect_img_path = "./dataset/training/s3/defect.tif"

normal_img = pil_loader(normal_img_path)
defect_img = pil_loader(defect_img_path)

#patch = rgb2gray(np.array(normal_img)[0:660, 0:100, :])
patch = rgb2gray(np.array(normal_img)[10:710, 10:710, :])
defect_img = rgb2gray(np.array(defect_img)[:, :, :])

result = match_template(defect_img, patch)
ij = np.unravel_index(np.argmax(result), result.shape)
print(ij)
x, y = ij[::-1]

a1, ax1 = plt.subplots(1)
plt.imshow(normal_img, cmap=plt.cm.gray)

hcoin, wcoin = patch.shape
rect1 = plt.Rectangle((0, 0), wcoin, hcoin, edgecolor='r', facecolor='none')
ax1.add_patch(rect1)
plt.show()

a2, ax2 = plt.subplots(1)
plt.imshow(defect_img, cmap=plt.cm.gray)
# highlight matched region
hcoin, wcoin = patch.shape
rect2 = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect2)
plt.show()

plt.imshow(result)
# highlight matched region
plt.autoscale(False)
plt.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()
