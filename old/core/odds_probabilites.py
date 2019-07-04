import numpy as np
from scipy.optimize import root, fsolve, brentq
import pandas as pd

def basic_method (probs):
    imp = list()
    for i in probs:
        imp.append(1/i)

    ap = []
    for i in imp:
        ap.append(i/sum(imp))
    return ap


def shin_func(x, imp):

    bb = sum(imp)
    ab = (np.sqrt(x**2 + 4*(1 - x) * (((imp)**2)/bb) ) - x) / (2*(1-x))
    return ab

def shin_for(x, imp):

    tmp = shin_func(x, imp)
    return 1 - sum(tmp)

odds = np.array([4.20, 3.70, 1.95])

imp = 1/odds

root = (brentq(
    shin_for, 0.0, 0.5, args=(imp)))

probs = shin_func(x=root, imp=imp)
print(probs)


