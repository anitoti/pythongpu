#!/usr/bin/env python3
# Run:
# 

################
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

mat = loadmat("/home/atotilca/pythongpu/data/DTI_A.mat")

# Grab the 83x83 matrix array

A = mat['A'] if 'A' in mat else next(v for k,v in mat.items() if not k.startswith('_'))

plt.figure(figsize=(6,6))
plt.imshow(A, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.title("DTI_A Structural Adjacency Matrix")
plt.savefig("/home/atotilca/pythongpu/data/dti_visual.png", dpi=200)