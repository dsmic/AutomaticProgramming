#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:22:15 2020

@author: detlef
"""

import numpy as np


def string_to_bin_narray(x):
    b = list(bin(abs(hash(x))))[2:]
    b = [0] * (63-len(b)) + b
    b = np.array(b, dtype = float) - 0.5
    return b

def string_to_simelar_has(x):
    h = np.array([0] * 63, dtype = float)
    if len(x) <= 3:
        return string_to_bin_narray(x)
    for i in range(len(x)-3):
        h += string_to_bin_narray(x[i:i+3])
    return h #/ (len(x)-3)




print(string_to_simelar_has("hallo was ist das"))
print(string_to_simelar_has("halo was ist das"))
print(string_to_simelar_has("hallo was it das"))
print(string_to_simelar_has("hal"))

print(string_to_simelar_has("hallo was ist dasd"))