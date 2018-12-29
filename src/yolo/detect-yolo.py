import os
from yolo import YOLO
from PIL import Image

def main():
    yolo = YOLO()

    imagepath = '../images/image2.jpg'
    
    image = Image.open(imagepath)

    r_image = yolo.detect_image(image)
    r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    main()