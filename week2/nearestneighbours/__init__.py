import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


def kNeighbourClass(k, features, classes):
    nfolds = 5
    kf = KFold(len(features), nfolds, shuffle=True, random_state=42)
    acc = []
    i = 0
    for train_index, test_index in kf:
        xTrain, xTest = features[train_index], features[test_index]
        yTrain, yTest = classes[train_index], classes[test_index]
        kClass = KNeighborsClassifier(k)
        kClass.fit(xTrain, yTrain)
        acc.append(kClass.score(xTest, yTest))
        i += 1
    meanAcc = np.mean(acc)
    return meanAcc


def bestkNN(krange, features, classes):
    kNeighRes = [kNeighbourClass(k, features, responses) for k in krange]
    for ind, res in zip(krange, kNeighRes):
        print 'k=%i: %f' % (ind, res)
    best = max(kNeighRes)
    best_ind = kNeighRes.index(best)
    bestK = krange[best_ind]
    print 'max accuracy: %f, (k=%i)' % (best, bestK)
    return (bestK, best)


with open('wine.data', 'r') as file:
    lines = file.readlines()
    data = [[float(p) for p in l.split(',')] for l in lines]
for row in data:
    row[0] = int(row[0])
features = np.array([row[1:len(row)] for row in data])
responses = np.array([row[0] for row in data])

compareKNN = bestkNN(xrange(1, 51), features, responses)
str1 = str(compareKNN[0])
str2 = str(compareKNN[1])
out('task1.txt', str1)
out('task2.txt', str2)

features_scaled = scale(features)
print '\nWith scaled features:'
compareKNN_scaled = bestkNN(xrange(1, 51), features_scaled, responses)
str1 = str(compareKNN_scaled[0])
str2 = str(compareKNN_scaled[1])
out('task3.txt', str1)
out('task4.txt', str2)
