import numpy as np
import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold


def out(filename, s):
    f = open(filename, 'w')
    f.write(s)
    f.close()


def estimate_metric(x, y, p):
    kf = KFold(len(x), n_folds=5, random_state=42)
    nNeighbors = 5
    kNeigh = KNeighborsRegressor(n_neighbors=nNeighbors, weights='distance', p=p, metric='minkowski')
    score = cross_val_score(kNeigh, x, y, scoring='mean_squared_error', cv=5)
    mean_score = np.mean(score)
    return mean_score


boston = datasets.load_boston()
results = boston.target
features_scaled = scale(boston.data)

pRange = np.linspace(1, 50, num=200)
scores = [estimate_metric(features_scaled, results, p) for p in pRange]

for (p, sc) in zip(pRange, scores):
    print 'p=%f, score=%f' % (p, sc)
best_score = max(scores)
best_p = pRange[scores.index(best_score)]
print 'best p=%f, score = %f' % (best_p, best_score)
result = '%.1f' % best_p
out('task1.txt', result)