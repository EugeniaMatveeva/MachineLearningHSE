import pandas
import re
from collections import Counter


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

df = pandas.read_csv('titanic.csv', index_col='PassengerId')

malesCnt = len(df[df['Sex'] == 'male'].index)
femalesCnt = len(df[df['Sex'] == 'female'].index)
res = '%s %s' % (malesCnt, femalesCnt)
out('1.txt', res)

res = '%.0f' % (len(df[df['Survived'] == 1].index/len(df.index)*100));
out('2.txt', res);

res = '%.2f' % (float(len(df[df['Pclass'] == 1].index))/len(df.index)*100);
out('3.txt', res);

res = '%.2f %.2f' % (df.mean(axis=0)['Age'], df.median(axis=0)['Age']);
out('4.txt', res);

res = '%.2f' % df.corr(method='pearson')['Parch']['SibSp'];
out('5.txt', res);


cnt = Counter();
fNames = df[df['Sex'] == 'female']['Name'].str.split(r'[" ,.()]+').apply(cnt.update);
del cnt['Miss']
del cnt['Mrs']
del cnt['']
res = cnt.most_common(1)[0][0];
out ('6.txt', cnt.most_common(1)[0][0])
