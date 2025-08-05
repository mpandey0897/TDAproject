import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection

print('hej')
# get working directory
import os
# print current working directory
print(os.getcwd())


# Load the data into a DataFrame
bunny = pd.read_csv(os.getcwd() + "/data/bunny.txt")

# Display the first few rows
print(bunny.head())
#size of bunny
print('bunny size:', bunny.shape)
#pick random m indices from 0 to 4000
idx = np.random.choice(bunny.shape[0], size=4000, replace=False)
print('idx:',idx)


print(bunny.iloc[idx])
bunny_sub = bunny.iloc[idx]
print(bunny_sub.head())


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


