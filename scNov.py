#!/usr/bin/python3
import os
import numpy as np # linear algebra
arq= 'window/fix10sel.csv'
lsl = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90] # pt
gama = 0.0025

for ls in lsl:
    cmd = 'python3 voteRoonSVM.py '+ arq +" "+ str(ls) +" "+ str(gama)
    #print(cmd)
    os.system(cmd)
