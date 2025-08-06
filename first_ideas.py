import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
import os
import gudhi
import math
from ripser import ripser
from persim import plot_diagrams
from persim import PersistenceImager
from gudhi.representations import Landscape as Landscape
from gudhi.representations import PersistenceImage as PersImage
from gudhi.subsampling import pick_n_random_points as pick_random

plt.rcParams['savefig.dpi'] = 300  # High-quality by default

# print current working directory
print(os.getcwd())


# Load the data into a DataFrame
bunny = pd.read_csv(os.getcwd() + "/data/bunny.txt")

# Display the first few rows
print(bunny.head())
#size of bunny
print('bunny size:', bunny.shape)

#pick random n indices
np.random.seed(42)
idx = np.random.choice(bunny.shape[0], size=400, replace=False)
print('idx:',idx, 'len(idx):', len(idx))

print(bunny.iloc[idx])
bunny_sub = bunny.iloc[idx]
print('bunny_sub:\n',bunny_sub.head())
#bunny_sub = bunny_sub[['x', 'y', 'z']].astype(float).to_numpy()
#print('bunny_sub:\n',bunny_sub)


# Convert DataFrame to float NumPy array for gudhi
bunny_points = bunny[['x', 'y', 'z']].astype(float).to_numpy()
print('bunny_points:\n', bunny_points)


#'''
# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


#ax.scatter(bunny['x'], bunny['y'], bunny['z'], s=1, c='blue', alpha=0.6)
ax.scatter(bunny_sub['x'], bunny_sub['y'], bunny_sub['z'], s=1, c='blue', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Cloud of Bunny Data n={len(idx)}')

plt.show()
#'''



ac = gudhi.AlphaComplex(points = bunny_points[idx]) #bunny_points
acx = ac.create_simplex_tree()
diag = acx.persistence()

n = len(idx)

# 2. Persistence Diagram
gudhi.plot_persistence_diagram(diag)
plt.title(f'Persistence Diagram of Bunny n={n}')
plt.savefig(f"Bunny_{n}_persistence_diagram.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Barcode Diagram (all dimensions)
gudhi.plot_persistence_barcode(diag)
plt.title(f'Barcode (Persistent Betti Numbers) of Bunny n={n}')
plt.savefig(f"Bunny_{n}_barcode.png", dpi=300, bbox_inches='tight')
plt.show()

# Filter persistence pairs to only dimension 2 (H2: voids), and finite intervals
diag_H2 = [
    (dim, pair) for dim, pair in diag
    if dim == 2 and not math.isinf(pair[1]) and pair[1] > pair[0]
]

print(f"Number of finite H2 intervals: {len(diag_H2)}")

# Plot barcodes for H2
if diag_H2:
    gudhi.plot_persistence_barcode(diag_H2)
    plt.title(f'Barcode of H2 (2-cycles) for Bunny n={len(idx)}')
    plt.savefig(f"Bunny_{n}_barcodeH2.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No finite H2 intervals to plot!")