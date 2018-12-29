import sys
sys.path.append('..')
from shared import *

IMG_PATH = '../../images'
ANNOT_PATH = '../../annotations'

labels = {
    'arrabida': 521,
    'camara': 452,
    'clerigos': 575,
    'musica': 325,
    'serralves': 205
}

keys = {
    'arrabida': 0,
    'camara': 1,
    'clerigos': 2,
    'musica': 3,
    'serralves': 4
}

def main():
    global IMG_PATH
    global ANNOT_PATH

    db = Database(IMG_PATH, ANNOT_PATH)

    f = open('labels.txt','w')

    for label in labels:
        for i in range(0, labels[label]):
            try:
                name = label + '-' + str(i).zfill(4)
                img_path = db.get_img_path(name)
                print('ola')
                xmin,xmax,ymin,ymax = db.annot_coords(name)
                print('boas')
                output = img_path + ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(keys[label])
                print(output)
            except:
                continue
    f.close()


if __name__ == "__main__":
    main()