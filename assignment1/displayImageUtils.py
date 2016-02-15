import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.ndimage
import math

def display_all(folder):
    image_files = os.listdir(folder)
    num_images = len(image_files)
    rows = int(math.ceil(math.sqrt(num_images)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    for i,img_name in enumerate(image_files):
        try:
            fname = os.path.join(folder,img_name)
            img = plt.imread(fname)
            im = ax.imshow(img)
            plt.draw()
            accept = raw_input('OK?')
        except IOError as e:
            print("skipping ")

display_all('notMNIST_small/A')
