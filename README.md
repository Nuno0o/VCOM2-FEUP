# VCOM2
VCOM landmarks

To run feature detection:
cd src/features
python feature-detection.py
flags:  -desc to generate descriptors
        -bow to generate bag of words
        -train to train svm
        -test to test svm on testing set
        -testone to iteratively test the model one by one

To run YOLO:
python vcom-detect-yolo.py
flags:  -i to iteratively test the model one by one