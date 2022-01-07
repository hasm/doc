#!/usr/bin/python3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import time
import sys
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import VarianceThreshold
from math import hypot

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
#ls = float(sys.argv[2])

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

x1,y1 = lof2(dfu,df['X'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1,y1, test_size=0.30, random_state=seed)

X_r, Y_r = SMOTE(k_neighbors=5, random_state=seed, n_jobs=4).fit_resample(X_train, Y_train)
model.fit(X_r, Y_r)



model.fit(X_r, Y_r)
 
acc = model.score(X_test, Y_test)
y_pred = model.predict_proba(X_test)
y_pred2 = model.predict(X_test)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))

