import cv2
import glob
import numpy as np
import os
import xml.etree.ElementTree as ET

class Database():
    def __init__(self, path_imgs, path_annot):
        self.path_imgs = path_imgs
        self.path_annot = path_annot

    # Get img path from name
    def get_img_path(self, name):
        return glob.glob(self.path_imgs + '/**/' + name + '.*', recursive=True)[0]
    
    # Get annotation path
    def get_annot_path(self, name):
        return glob.glob(self.path_annot + '/**/' + name + '.xml', recursive=True)[0]

    # Use name like 'arrabida-0001'
    def annot_coords(self, name):
        print('here')
        annotPath = get_annot_path(name)
        print('xd')
        try:
            tree = ET.parse(annotPath)
            root = tree.getroot()
            branch = root.find('object').find('bndbox')
            xmin = int(branch.find('xmin').text)
            xmax = int(branch.find('xmax').text)
            ymin = int(branch.find('ymin').text)
            ymax = int(branch.find('ymax').text)
            return (xmin,xmax,ymin,ymax)
        except:
            return (0,0,0,0)
    
    # Use name like 'arrabida-0001'
    def read_img(self, name):
        imagePath = get_img_path(name)
        img = cv2.imread(imagePath)
        return img
    
    # Get image with annotated region
    def read_img_region(self, name):
        img = self.read_img(name)
        xmin,xmax,ymin,ymax = self.annot_coords(name)
        if xmin == xmax and xmax == ymin and ymin == ymax and ymax == 0:
            return img
        else:
            try:
                img2 = img_region(img,xmin,xmax,ymin,ymax)
                return img2
            except:
                return img

# Extract image region
def img_region(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax,xmin:xmax]

# Get full path
def get_full_path(path):
    return os.path.abspath(path)

if __name__ == "__main__":
    pass