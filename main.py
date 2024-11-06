import cv2
import matplotlib.pyplot as plt
import read_files
from os import walk

path = "Stare/images"
path_to = "Stare/mask"

f = []

for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break
    
for filename in f:
    image = cv2.imread(path + "/" + filename)[..., 1].astype("float32")
    image = (image - image.min())/(image.max() - image.min())
    print(image.max(), image.min())
    image[image>=0.2] = 1
    cv2.imwrite(path_to + "/" + str(filename).split(".")[0]+".jpg", image)
    plt.imshow(image, cmap='gray')
    plt.title(filename)
    plt.show()
    