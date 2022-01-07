#!/usr/bin/python3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import time
import sys
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import VarianceThreshold
from math import hypot
from sklearn import svm

test_size = 0.30
seed = 42
outliers_fraction = 0.007

def distPoint(gw,pt):
   #print(gw['x_pt'].str.replace(',','.').item())
   x1 = gw['x_roon'].str.replace(',','.').item()
   x2 = pt['x_roon'].str.replace(',','.').item()
   y1 = gw['y_roon'].str.replace(',','.').item()
   y2 = pt['y_roon'].str.replace(',','.').item()
   xf = float(x1) - float(x2)
   yf = float(y1) - float(y2)
   return hypot(xf,yf)


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

rF = pd.read_csv('roonsF.csv')

# new data frame with split value columns 
new = df['LABEL'].str.split("_", n = 3, expand = True) 
  
# making seperate first name column from new data frame 
df['X']= new[0] 
  
# making seperate last name column from new data frame 
df['Y']= new[1]

# making seperate last name column from new data frame 
df['Z']= new[2]  


X = np.array(dfu)
y = np.array(df['X']) # another way of indexing a pandas df

roons = df['X'].value_counts().sort_index().index

##### pre processing

###########MUlt models

model = KNeighborsClassifier(n_neighbors=9)

#X_train, X_test, Y_train, Y_test = train_test_eq_split(dfu,df['LABEL'],29,seed)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dfu,df['X'], test_size=0.30, random_state=seed)

X_r, Y_r = SMOTE(k_neighbors=5, random_state=seed, n_jobs=4).fit_resample(X_train, Y_train)
model.fit(X_r, Y_r)


uy = np.unique(Y_r)
dicts = {}
keys = uy
for x in uy : 
  e = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=klof).fit(X_r[Y_r == x])
  dicts[x] = e


model.fit(X_r, Y_r)
 
uy = np.unique(Y_test)
#y_pred = model.predict(X_test)

for x in uy :
    print("C:", x)
    rmp = []
    probas = []
    cont = 0
    hit = 0
    hit2 = 0
    index = 0
    index2 = Y_test[Y_test == x].index
    index3 = X_test[Y_test == x].index
    erro = []

    y_pred = model.predict_proba(X_test[Y_test == x])

    for row in y_pred :
        maioresi = row.argsort()[-3:][::-1]
        maiores = [row[maioresi[0]],row[maioresi[1]], row[maioresi[2]]]
        probas.append(float(row[maioresi[0]]))
        rmp.append(int(roons[maioresi[0]]))
        n1 = dicts[roons[maioresi[0]]].predict(X_test.iloc[index, :].values.reshape(1, -1)) # testar oss 3 maiores?
        n2 = dicts[roons[maioresi[1]]].predict(X_test.iloc[index, :].values.reshape(1, -1))
        n3 = dicts[roons[maioresi[2]]].predict(X_test.iloc[index, :].values.reshape(1, -1))
        if float(row[maioresi[0]]) > ls :
            cont+=1
            if roons[maioresi[0]] == Y_test[index2[index]]:
                hit+=1
            r1 = roons[maioresi[0]]
            r2 = Y_test[index2[index]]
            pontoE = rF.loc[rF['R'] == int(r1)]
            ponto1 = rF.loc[rF['R'] == int(r2)] 
            erro.append(distPoint(pontoE,ponto1))
        else:
            if n1 == 1 and float(row[maioresi[0]]) > lf:
                cont+=1
                if roons[maioresi[0]] == Y_test[index2[index]]:
                    hit+=1
                    hit2+=1
                r1 = roons[maioresi[0]]
                r2 = Y_test[index2[index]]
                pontoE = rF.loc[rF['R'] == int(r1)]
                ponto1 = rF.loc[rF['R'] == int(r2)] 
                erro.append(distPoint(pontoE,ponto1))
            elif n2 == 1 and float(row[maioresi[1]]) > lf:
                cont+=1
                if roons[maioresi[1]] == Y_test[index2[index]]:
                    hit+=1
                    hit2+=1
                r1 = roons[maioresi[1]]
                r2 = Y_test[index2[index]]
                pontoE = rF.loc[rF['R'] == int(r1)]
                ponto1 = rF.loc[rF['R'] == int(r2)] 
                erro.append(distPoint(pontoE,ponto1))
            elif n3 == 1 and float(row[maioresi[2]]) > lf:
                cont+=1
                if roons[maioresi[2]] == Y_test[index2[index]]:
                    hit+=1
                    hit2+=1
                r1 = roons[maioresi[2]]
                r2 = Y_test[index2[index]]
                pontoE = rF.loc[rF['R'] == int(r1)]
                ponto1 = rF.loc[rF['R'] == int(r2)] 
                erro.append(distPoint(pontoE,ponto1))
        index+=1
    err = pd.DataFrame(erro).mean()
    print(ls, lf, hit, cont, hit/cont, err[0], cont/len(y_pred),len(y_pred) )
