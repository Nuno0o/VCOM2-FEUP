import cv2
import glob
import numpy as np
import os
import xml.etree.ElementTree as ET

class Database():
    def __init__(self, path_imgs, path_annot):
        self.path_imgs = path_imgs
        self.path_annot = path_annot

    # Use name like 'arrabida-0001'
    def annot_coords(self, name):
        annotPath = glob.glob(self.path_annot + '/*/' + name + '.xml')[0]
        tree = ET.parse(annotPath)
        xmin = int(tree.find('xmin').text)
        xmax = int(tree.find('xmax').text)
        ymin = int(tree.find('ymin').text)
        ymax = int(tree.find('ymax').text)
        return (xmin,xmax,ymin,ymax)
    
    # Use name like 'arrabida-0001'
    def read_img(self, name):
        imagePath = glob.glob(self.path_imgs + '/*/' + name + '.*')[0]
        img = cv2.imread(imagePath)
        return img

# Extract image region
 def img_region(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax,xmin:xmax]

if __name__ == "__main__":
    pass