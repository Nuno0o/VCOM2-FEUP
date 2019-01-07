import sys
sys.path.append('..')

from shared import *
import argparse
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='VCOM - Feature Image Classifier for Landmarks')

parser.add_argument('-desc', action='store_true', help='Compute image descriptors')

parser.add_argument('-bow', action='store_true', help='Compute Bag of Words')

parser.add_argument('-train', action='store_true', help='Train classifier')

parser.add_argument('-test', action='store_true', help='Test classifier')

parser.add_argument('-testone', action='store_true', help='Test classifier for one image at a time')

args = parser.parse_args()

DESC = args.desc

BOW = args.bow

TRAIN = args.train

TEST = args.test

TESTONE = args.testone

IMG_PATH = '../../images'
ANNOT_PATH = '../../annotations'

DESC_PATH = 'descriptors.pkl'
BOW_PATH = 'bow.pkl'
TRAIN_PATH = 'model.pkl'
TRAIN2_PATH = 'train.pkl'

db = Database(IMG_PATH, ANNOT_PATH)

labels = {
    'arrabida': 416,#521
    'camara': 361,#452
    'clerigos': 460,#575
    'musica': 260,#325
    'serralves': 164#,#205
    #'none': 240
}

labels_test = {
    'arrabida': 521,
    'camara': 452,
    'clerigos': 575,
    'musica': 325,
    'serralves': 205#,
    #'none': 299
}

keys = {
    'arrabida': 0,
    'camara': 1,
    'clerigos': 2,
    'musica': 3,
    'serralves': 4#,
    #'none': 5
}

def main():
    global db

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
        bow = bow_trainer(100)
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

    svm = None

    if TRAIN:
        print('Starting train...')
        extractor = bow_extractor(dictionary)
        svm = create_svm()

        X = []
        y = []

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
                    X.append(bow_extract(extractor, img, features[0])[0])
                    y.append(keys[label])
                except Exception as e:
                    continue
        
        print('Training model...')
        train_svm(svm,X,y)
        print('Saving model...')
        store_object([X,y], TRAIN2_PATH)
        store_svm(svm, TRAIN_PATH)
        print('Done')
    else:
        try:
            print('Loading model...')
            svm = load_svm(TRAIN_PATH)
            print('Done')
        except:
            print('Cant load model file, run again with -train to create a new one')
            quit()
    if TEST:
        extractor = bow_extractor(dictionary)
        bows = []
        trues = []
        for label in labels:
            for i in range(labels[label], labels_test[label]):
                try:
                    print('Extracting bow for ' + str(label) + ' ' + str(i) + '/' + str(labels_test[label]))
                    name = label + '-' + str(i).zfill(4)
                    img = db.read_img(name)
                    img = gray(img)
                    img = resize_img(img)
                    features = get_key_points(img)
                    bows.append(bow_extract(extractor, img, features[0])[0])
                    trues.append(keys[label])
                except Exception as e:
                    continue
        pred = np.squeeze(test_svm(svm, bows)[1].astype(int))
        print('Confusion Matrix: \n' + str(confusion_matrix(trues,pred)))
        correct = 0
        for i in range(0, len(trues)):
            if trues[i] == pred[i]:
                correct += 1
        print('Accuracy: ' + str(correct/len(trues)))
    if TESTONE:
        print('Write image names to predict, \'exit\' to close')
        extractor = bow_extractor(dictionary)
        while True:
            image_name = input('Insert image name(e.g. \'arrabida-0000\'): ')
            if image_name == 'exit':
                print('Leaving...')
                exit()
            bows = []
            trues = []
            print('Extracting bow for ' + image_name)
            try:
                img = db.read_img(image_name)
            except:
                print('Error reading image, try again')
                continue
            img = gray(img)
            img = resize_img(img)
            features = get_key_points(img)
            bows.append(bow_extract(extractor, img, features[0])[0])
            trues.append(keys[image_name.split('-')[0]])
            pred = np.squeeze(test_svm(svm, bows)[1].astype(int))
            print('Predicted ' + str(pred) + ' , was ' + str(trues[0]))


        


    



if __name__ == "__main__":
    main()