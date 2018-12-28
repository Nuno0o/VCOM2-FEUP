from shared import *

IMG_PATH = '../images'
ANNOT_PATH = '../annotations'

def main():
    global IMG_PATH
    global ANNOT_PATH

    db = Database(IMG_PATH, ANNOT_PATH)

    img = db.read_img_region('arrabida-0000')


if __name__ == "__main__":
    main()