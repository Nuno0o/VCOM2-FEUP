import cv2
import glob
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import imutils
from sklearn import svm

WIDTH = 512

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
        try:
            annotPath = self.get_annot_path(name)
            tree = ET.parse(annotPath)
            root = tree.getroot()
            branch = root.find('object').find('bndbox')
            xmin = round(float(branch.find('xmin').text))
            xmax = round(float(branch.find('xmax').text))
            ymin = round(float(branch.find('ymin').text))
            ymax = round(float(branch.find('ymax').text))
            return (xmin,xmax,ymin,ymax)
        except:
            return (0,0,0,0)
    
    # Use name like 'arrabida-0001'
    def read_img(self, name):
        imagePath = self.get_img_path(name)
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

# Get SIFT features and descriptors
def get_key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return (kp, des)

# Match images
def match_features(features1, features2):
    des1 = features1[1]
    des2 = features2[1]

    fb = cv2.BFMatcher()

    matches = fb.match(des1, des2)

    matches = sorted(matches, key = lambda x: x.distance)
    
    return matches

def bow_trainer(dictionary_size):
    return cv2.BOWKMeansTrainer(50)

def bow_cluster(trainer, desc):
    return trainer.cluster(desc)

def bow_extractor(dictionary):
    detector = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher()
    extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
    extractor.setVocabulary(dictionary)
    return extractor

def bow_extract(extractor, img, desc):
    return extractor.compute(img, desc)

def create_svm():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return svm

def train_svm(svm,X,y):
    svm.trainAuto(array_to_np(X), cv2.ml.ROW_SAMPLE, array_to_np(y), kFold=15)

def test_svm(svm, pred):
    return svm.predict(np.array(pred))

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def store_object(desc_list, path):
    pickle.dump(desc_list, open(path, 'wb'))

def load_object(path):
    return pickle.load(open(path, 'rb'))

def store_svm(svm, path):
    svm.save(path)

def load_svm(path):
    return cv2.ml.SVM_load(path)


def resize_img(img):
    return imutils.resize(img, width=WIDTH)

def array_to_np(arr):
    return np.array(arr)


if __name__ == "__main__":
    pass