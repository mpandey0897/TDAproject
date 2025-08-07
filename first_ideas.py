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
idx = np.random.choice(bunny.shape[0], size=4000, replace=False) #35947
print('idx:',idx, 'len(idx):', len(idx))

print(bunny.iloc[idx])
bunny_sub = bunny.iloc[idx]
print('bunny_sub:\n',bunny_sub.head())
#bunny_sub = bunny_sub[['x', 'y', 'z']].astype(float).to_numpy()
#print('bunny_sub:\n',bunny_sub)


# Convert DataFrame to float NumPy array for gudhi
bunny_points = bunny[['x', 'y', 'z']].astype(float).to_numpy()
print('type of bunny_points:', type(bunny_points))
print('bunny_points:\n', bunny_points)

'''
#find minimal distance between two points
def min_distance(points):
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist
min_dist = min_distance(bunny_points[0:100])
print('min_dist:', min_dist)  # Check first 1000 points for speed
print('check:')
'''

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

# Convert to DataFrame for convenience
birth_death_data = [
    {'dimension': dim, 'birth': birth, 'death': death}
    for dim, (birth, death) in diag
    if not math.isinf(death)  # Optional: exclude infinite deaths
]

df_pd = pd.DataFrame(birth_death_data)
print(df_pd.head())
print('shape:\n',df_pd.shape)
#filter out dimension 0 this excludes 0 cycles (connected components)
df_pd = df_pd[df_pd['dimension'] > 0]
df_pdH1 = df_pd[df_pd['dimension'] == 1]
df_pdH2 = df_pd[df_pd['dimension'] == 2]
#two death divided by birth histograms for H1 and H2
plt.figure(figsize=(8, 6))
plt.hist(np.log(np.log(df_pdH1['death'] / df_pdH1['birth'])), bins=100, alpha=0.7, color='blue', edgecolor='black', label='H1')
plt.hist(np.log(np.log(df_pdH2['death'] / df_pdH2['birth'])), bins=100, alpha=0.7, color='orange', edgecolor='black', label='H2')
plt.xlabel('log(log(Death/Birth))')
plt.ylabel('Frequency')
plt.title(f'log(log(Death/Birth)) Histogram of Bunny n={n}')
plt.xlim(-20, 5)
plt.legend()
plt.grid(True)
plt.savefig(f"Bunny_{n}_loglog_death_birth_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

print(df_pd.head())
print('shape:\n',df_pd.shape)

# filter out pairs with birth below threshold eps= 0.0001
#eps = 0.0000001
#df_pd = df_pd[df_pd['birth'] >= eps]
#print('shape after filtering:\n',df_pd.shape)

#birth divided by death histogram
plt.figure(figsize=(8, 6))
plt.hist(np.log(np.log(df_pd['death'] / df_pd['birth'])), bins=100, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Birth/Death Ratio')
plt.ylabel('Frequency')
plt.title(f'log(log(Death/Birth)) Ratio Histogram of Bunny n={n}')
plt.xlim(-20, 5)
plt.grid(True)
plt.savefig(f"Bunny_{n}_loglog_death_birth_ratio_histogram.png", dpi=300, bbox_inches='tight')
plt.show()


# scatter plot of persistence pairs
plt.figure(figsize=(8, 6))
plt.scatter(df_pd['birth'], df_pd['death'], alpha=0.5, s=10)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title(f'Persistence Pairs of Bunny n={n}')
plt.xlim(0, df_pd['birth'].max() * 1.1)
plt.ylim(0, df_pd['death'].max() * 1.1)
plt.grid(True)
plt.savefig(f"Bunny_{n}_persistence_pairs.png", dpi=300, bbox_inches='tight')
plt.show()


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
    
