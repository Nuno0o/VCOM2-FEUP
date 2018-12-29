import os
import sys

from yolo import YOLO
from PIL import Image

def main():
    yolo = YOLO()

    imagepath = '../../../images/clerigos/clerigos-0039.jpg'
    
    image = Image.open(imagepath)

    r_image = yolo.detect_image(image)
    r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    main()