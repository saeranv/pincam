# import sys
# import os

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
import functools as ft
from pprint import pprint
import matplotlib.pyplot as plt

# Set pandas option
pd.set_option('precision', 2) # will also set for gpd
pp = lambda x, *args: pprint(x) if not args else print(x,*args)
ppln = lambda v: print(v, sep='\n')
ppt = lambda v: pp(type(v))


def is_near_zero(val, eps=1e-10):
    return abs(val) < eps

def fd(module, key=None):
    """ To efficiently search modules in osm"""
    hfd = lambda m, k: k.lower() in m.lower()
    if key is None:
        return [m for m in dir(module)][::-1]
    else:
        return [m for m in dir(module) if hfd(m, key)][::-1]
