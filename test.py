
from collections import namedtuple
from datetime import datetime
from functools import partial
from sklearn import ensemble, datasets
import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
import json
import time

plat = lambda: "PyPy"
import platform
platform.python_implementation = plat
from compiledtrees.compiled import CompiledClassifierPredictor

SAMPLES = 10000
d = datasets.make_hastie_10_2(n_samples=SAMPLES)
clf = ensemble.RandomForestClassifier(n_estimators=3, max_depth=5)
# clf = ensemble.RandomForestClassifier(n_estimators=50, max_depth=15)
X, y = d
clf.fit(X, y)
print("sklearn-trees:")
ts = time.time()
# for d in X:
    # res = clf.predict_proba([d])
res = clf.predict_proba(X)
print(time.time()-ts, "seconds")
print(res[-1])
clf = "/tmp/compiledtrees_az3m33u7.so"
cclf = CompiledClassifierPredictor(clf)
X2 = X.tolist()
X3 = []
# for x in X2:
    # X3.append([j*1000 for j in x])
X3 = X2
# print("X3", X3[0])
print("hyperc-trees:")
ts = time.time()
for d in X3:
    # res = cclf.predict_proba([d], _output=probas2)
    res = cclf.predict_proba(d)
print(time.time()-ts, "seconds")
print(res[0], res[1])
# print(res)
