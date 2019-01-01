import numpy as np 
from sklearn.metrics import confusion_matrix
keys = {
    'arrabida': 0,
    'camara': 1,
    'clerigos': 2,
    'musica': 3,
    'serralves': 4,
    'none': 5
}

FILE = 'true3.txt'

f = open(FILE,'r')

preds = f.readlines()

f.close()

ncorrect = 0

true = []
pred = []
area_acc = []

for line in preds:
    content = line.split(',')
    prediction = content[1].split(' ')[0]
    if prediction == '0':
        prediction = 'none'
    true.append(keys[content[0]])
    pred.append(keys[prediction])
    if prediction == content[0]:
        ncorrect += 1
        area_true = (int(content[3]) - int(content[2])) * (int(content[5]) - int(content[4]))
        area_pred = (int(content[7]) - int(content[6])) * (int(content[9]) - int(content[8]))
        area_int = (min(int(content[3]), int(content[7]))-max(int(content[2]), int(content[6]))) * (min(int(content[5]), int(content[9]))-max(int(content[4]),int(content[8])))
        if prediction != 'none':
            if area_int > 0:
                area_acc.append(area_int/(area_true+area_pred-area_int))
            else:
                area_acc.append(0)


print('Accuracy: ' + str(ncorrect/len(preds)))

print(str(confusion_matrix(true,pred)))

print('Region accuracy: ' + str(sum(area_acc)/ float(len(area_acc))))