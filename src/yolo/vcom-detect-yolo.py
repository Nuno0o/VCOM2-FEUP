import os
import sys
import argparse
sys.path.append('..')

from shared import *
from sklearn.metrics import confusion_matrix
from yolo import YOLO
from PIL import Image

parser = argparse.ArgumentParser(description='VCOM - YOLO Image Classifier for Landmarks')

parser.add_argument('-i', action='store_true', help='Choose images to test interatively')

args = parser.parse_args()

INT = args.i

IMG_PATH = '../../images'
ANNOT_PATH = '../../annotations'

TRUE = 'true.txt'
PRED = 'pred.txt'

db = Database(IMG_PATH, ANNOT_PATH)

labels = {
    'arrabida': 416,#521
    'camara': 361,#452
    'clerigos': 460,#575
    'musica': 260,#325
    'serralves': 164,#,#205
    'none': 240
}

labels_test = {
    'arrabida': 521,
    'camara': 452,
    'clerigos': 575,
    'musica': 325,
    'serralves': 205,#,
    'none': 299
}

keys = {
    'arrabida': 0,
    'camara': 1,
    'clerigos': 2,
    'musica': 3,
    'serralves': 4,#,
    'none': 5
}

def main():
    global db

    yolo = YOLO(model_path='model_data/yolo_vcom.h5',anchors_path='model_data/yolo_anchors.txt',classes_path='model_data/vcom_classes.txt')

    if not INT:

        file = open(TRUE, 'w')

        for label in labels:
            for i in range(labels[label], labels_test[label]):
                try:
                    print('Classifying ' + str(label) + ' ' + str(i) + '/' + str(labels_test[label]))
                    sys.stdout.flush()
                    name = label + '-' + str(i).zfill(4)
                    path = db.get_img_path(name)
                    img = Image.open(path)
                    xmin,xmax,ymin,ymax = db.annot_coords(name)
                    _, pxmin,pxmax,pymin,pymax,score,pred = yolo.detect_image(img)
                    file.write(str(label)+','+str(pred)+','+str(xmin)+','+str(xmax)+','+str(ymin)+','+str(ymax)+','+str(pxmin)+','+str(pxmax)+','+str(pymin)+','+str(pymax)+'\n')
                    file.flush()
                except Exception as e:
                    print(e)
                    continue

        file.close()
    
    else:
        while True:
            image_name = input('Insert image name(e.g. \'arrabida-0000\'): ')
            if image_name == 'exit':
                print('Leaving...')
                exit()
            try:
                path = db.get_img_path(image_name)
            except:
                print('Error reading image, try again')
                continue
            img = Image.open(path)
            xmin,xmax,ymin,ymax = db.annot_coords(image_name)
            detect, pxmin,pxmax,pymin,pymax,score,pred = yolo.detect_image(img)
            detect.show()

    yolo.close_session()

if __name__ == '__main__':
    main()