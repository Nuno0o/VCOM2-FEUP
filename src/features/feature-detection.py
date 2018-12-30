import sys
sys.path.append('..')

from shared import *
import argparse

parser = argparse.ArgumentParser(description='VCOM - Feature Image Classifier for Landmarks')

parser.add_argument('-desc', action='store_true', help='Compute image descriptors')

parser.add_argument('-bow', action='store_true', help='Compute Bag of Words')

parser.add_argument('-train', action='store_true', help='Train classifier')

args = parser.parse_args()

DESC = args.desc

BOW = args.bow

TRAIN = args.train

IMG_PATH = '../../images'
ANNOT_PATH = '../../annotations'

DESC_PATH = 'descriptors.pkl'
BOW_PATH = 'bow.pkl'

db = Database(IMG_PATH, ANNOT_PATH)

labels = {
    'arrabida': 416,#521
    'camara': 361,#452
    'clerigos': 460,#575
    'musica': 260,#325
    'serralves': 164#205
}

keys = {
    'arrabida': 0,
    'camara': 1,
    'clerigos': 2,
    'musica': 3,
    'serralves': 4
}

def main():
    global db
    global DESC
    global BOW
    global TRAIN

    all_descriptors = []

    if DESC:
        for label in labels:
            for i in range(0, labels[label]):
                try:
                    print('Computing descriptors for ' + str(label) + ' ' + str(i) + '/' + str(labels[label]))
                    sys.stdout.flush()
                    name = label + '-' + str(i).zfill(4)
                    img = db.read_img(name)
                    img = gray(img)
                    img = resize_img(img)
                    features = get_key_points(img)
                    all_descriptors.extend(features[1])
                except:
                    continue
        print('Storing descriptors...')
        store_object(all_descriptors, DESC_PATH)
    else:
        try:
            print('Loading descriptors...')
            all_descriptors = load_object(DESC_PATH)
            print('Done')
        except Exception as e:
            print('Cant load descriptors file, run again with -desc to create a new one')
            quit()
    
    all_descriptors = array_to_np(all_descriptors)
    dictionary = None

    if BOW:
        print('Computing bag of words...')
        bow = bow_trainer(50)
        dictionary = bow_cluster(bow, all_descriptors)
        store_object(dictionary, BOW_PATH)
    else:
        try:
            print('Loading dictionary...')
            dictionary = load_object(BOW_PATH)
            print('Done')
        except:
            print('Cant load dictionary file, run again with -bow to create a new one')
            quit()

    if TRAIN:
        print('Starting train...')
        extractor = bow_extractor(dictionary)

        for label in labels:
            for i in range(0, labels[label]):
                try:
                    print('Extracting bow for ' + str(label) + ' ' + str(i) + '/' + str(labels[label]))
                    sys.stdout.flush()
                    name = label + '-' + str(i).zfill(4)
                    img = db.read_img(name)
                    img = gray(img)
                    img = resize_img(img)
                    features = get_key_points(img)
                    print(bow_extract(extractor, img, features[0]))
                except Exception as e:
                    continue
        


    



if __name__ == "__main__":
    main()