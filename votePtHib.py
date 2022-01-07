#!/usr/bin/python3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import LocalOutlierFactor
import time
import sys
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import VarianceThreshold
from math import hypot
from sklearn import svm

def dist(x1,x2,y1,y2):
   xf = int(x1) - int(x2)
   yf = int(y1) - int(y2)
   return hypot(xf,yf)

def distPoint(gw,pt):
   #print(gw['x_pt'].str.replace(',','.').item())
   x1 = gw['x_pt'].str.replace(',','.').item()
   x2 = pt['x_pt'].str.replace(',','.').item()
   y1 = gw['y_pt'].str.replace(',','.').item()
   y2 = pt['y_pt'].str.replace(',','.').item()
   xf = float(x1) - float(x2)
   yf = float(y1) - float(y2)
   return hypot(xf,yf)

test_size = 0.30
seed = 42
outliers_fraction = 0.007


def train_test_eq_split(X, y, n_per_class, random_state=None):
    if random_state:
        np.random.seed(random_state)
    sampled = X.groupby(y, sort=False).apply(
        lambda frame: frame.sample(n_per_class))
    mask = sampled.index.get_level_values(1)

    X_train = X.drop(mask)
    X_test = X.loc[mask]
    y_train = y.drop(mask)
    y_test = y.loc[mask]

    return X_train, X_test, y_train, y_test


def lof2(x,y, kl=5):
   model = LocalOutlierFactor(n_neighbors=5, contamination=outliers_fraction)
   y_pred = model.fit_predict(x)
   #data["outlier"] = pd.DataFrame(y_pred)
   data = x.drop(x[y_pred < 0].index)
   data2 = y.drop(y[y_pred < 0].index)
   return data, data2


###########fetche data

arq = sys.argv[1]

ls = float(sys.argv[2])
lf = float(sys.argv[3])
klof = float(sys.argv[4])

df = pd.read_csv(arq)
dfu = df.iloc[:, 0:42]
dfu=dfu.replace(100,-105)
dfu.head()

pt = pd.read_csv('pointsF.csv')

X = np.array(dfu)
#y = np.array(df['X']) # another way of indexing a pandas df

roons = df['LABEL'].value_counts().sort_index().index

model = KNeighborsClassifier(n_neighbors=9)

#X_train, X_test, Y_train, Y_test = train_test_eq_split(dfu,df['LABEL'],29,seed)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dfu,df['LABEL'], test_size=0.30, random_state=seed)

X_r, Y_r = SMOTE(k_neighbors=5, random_state=seed, n_jobs=4).fit_resample(X_train, Y_train)
model.fit(X_r, Y_r)


uy = np.unique(Y_r)
dicts = {}
keys = uy
for x in uy : 
  e = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=klof).fit(X_r[Y_r == x])
  dicts[x] = e


model.fit(X_r, Y_r)
 
acc = model.score(X_test, Y_test)
y_pred = model.predict_proba(X_test)
y_pred2 = model.predict(X_test)
#print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))

rmp = []
probas = []
cont = 0
hit = 0
hit2 = 0
index = 0
index2 = Y_test.index
index3 = X_test.index
erro = []

for row in y_pred :
    maioresi = row.argsort()[-3:][::-1]
    maiores = [row[maioresi[0]],row[maioresi[1]], row[maioresi[2]]]
    probas.append(float(row[maioresi[0]]))
    rmp.append(int(roons[maioresi[0]]))
    n1 = dicts[roons[maioresi[0]]].predict(X_test.iloc[index, :].values.reshape(1, -1)) # testar oss 3 maiores?
    n2 = dicts[roons[maioresi[1]]].predict(X_test.iloc[index, :].values.reshape(1, -1))
    n3 = dicts[roons[maioresi[2]]].predict(X_test.iloc[index, :].values.reshape(1, -1))
    r = -1
    if float(row[maioresi[0]]) > ls:  # fazer um OU 1 ou outro proba ou novelty
        cont+=1
        if roons[maioresi[0]] == Y_test[index2[index]]:
            hit+=1
        r = roons[maioresi[0]]
    else:
        if n1 == 1 and float(row[maioresi[0]]) > lf:
            cont+=1
            if roons[maioresi[0]] == Y_test[index2[index]]:
                hit+=1
                hit2+=1
            r = roons[maioresi[0]]
        elif n2 == 1 and float(row[maioresi[1]]) > lf:
            cont+=1
            if roons[maioresi[1]] == Y_test[index2[index]]:
                hit+=1
                hit2+=1
            r = roons[maioresi[1]]
        elif n3 == 1 and float(row[maioresi[2]]) > lf:
            cont+=1
            if roons[maioresi[2]] == Y_test[index2[index]]:
                hit+=1
                hit2+=1
            r = roons[maioresi[2]]
    if r != -1:
        rv = Y_test[index2[index]]
        new = r.split("_")  
        # making seperate first name column from new data frame 
        r1 = new[0] 
        # making seperate last name column from new data frame 
        x1 = new[1]
        # making seperate last name column from new data frame 
        y1 = new[2]
        new2 = rv.split("_")  
        # making seperate first name column from new data frame 
        r2 = new2[0] 
        # making seperate last name column from new data frame 
        x2 = new2[1]
        # making seperate last name column from new data frame 
        y2 = new2[2]  
          
        pontoE = pt.loc[((pt['R'] == int(r1)) & (pt['X'] == int(x1)) & (pt['Y'] == int(y1)) ) ]
        ponto1 = pt.loc[((pt['R'] == int(r2)) & (pt['X'] == int(x2)) & (pt['Y'] == int(y2)) ) ] 
        erro.append(distPoint(pontoE,ponto1))

    index+=1

err = pd.DataFrame(erro).mean()
print(ls, lf, hit, cont, hit/cont, err[0], cont/len(y_pred),len(y_pred) )
#print(erro)
np.savetxt("hib_"+str(int(ls*100))+'_'+str(int(lf*100))+'.csv',erro, delimiter=",", fmt="%s")

