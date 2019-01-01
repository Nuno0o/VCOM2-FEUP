import cv2 as cv
import numpy as np

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image


def main():
    ##################################################
    #detector = cv.xfeatures2d.SIFT_create()
    detector = cv.KAZE_create()
    matcher = cv.FlannBasedMatcher()
    bowTrainer = cv.BOWKMeansTrainer(100)
    bowExtractor = cv.BOWImgDescriptorExtractor(detector, matcher)
    ##################################################

    ## TODO: implement the rest of the code...

if __name__ == '__main__':
    main()
