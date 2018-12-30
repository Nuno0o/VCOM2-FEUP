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

    #for label in labels:
    #    for i in range(0, labels[label]):
    #        try:
    #            name = label + '-' + str(i).zfill(4)
    #            #img_path = get_full_path(db.get_img_path(name))
    #            #xmin,xmax,ymin,ymax = db.annot_coords(name)
    #            #output = img_path + ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(keys[label]) + '\n'
    #            
    #        except:
    #            continue
    img1 = db.read_img_region('arrabida-0010')
    img2 = db.read_img('arrabida-0011')

    img1 = gray(img1)
    img2 = gray(img2)

    fea1 = get_key_points(img1)
    fea2 = get_key_points(img2)

    matches = match_features(fea1, fea2)

    img3 = cv2.drawMatches(img1, fea1[0], img2, fea2[0], matches, None)

    cv2.imshow('img',img3)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()