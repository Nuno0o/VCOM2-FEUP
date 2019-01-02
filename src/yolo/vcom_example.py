import os
import sys

from yolo import YOLO
from PIL import Image

def main():
    yolo = YOLO(model_path='model_data/yolo_vcom.h5',anchors_path='model_data/yolo_anchors.txt',classes_path='model_data/vcom_classes.txt')

    imagepath = '../../images/arrabida/arrabida-0033.jpg'
    
    image = Image.open(imagepath)

    r_image,_,_,_,_,_,_ = yolo.detect_image(image)
    r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    main()