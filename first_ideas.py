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



# print current working directory
print(os.getcwd())


# Load the data into a DataFrame
bunny = pd.read_csv(os.getcwd() + "/data/bunny.txt")

# Display the first few rows
print(bunny.head())
#size of bunny
print('bunny size:', bunny.shape)
#pick random m indices from 0 to 4000
idx = np.random.choice(bunny.shape[0], size=40, replace=False)
print('idx:',idx)


print(bunny.iloc[idx])
bunny_sub = bunny.iloc[idx]
print('bunny_sub:\n',bunny_sub.head())
# Convert DataFrame to float NumPy array for gudhi
bunny_points = bunny[['x', 'y', 'z']].astype(float).to_numpy()
print('bunny_points:\n',bunny_points)

'''
# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


#ax.scatter(bunny['x'], bunny['y'], bunny['z'], s=1, c='blue', alpha=0.6)
ax.scatter(bunny_sub['x'], bunny_sub['y'], bunny_sub['z'], s=1, c='blue', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Bunny Data')

plt.show()
'''

ac = gudhi.AlphaComplex(points = bunny_points)
acx = ac.create_simplex_tree()
diag = acx.persistence()
gudhi.plot_persistence_diagram(diag)
plt.show()
