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

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(bunny['x'], bunny['y'], bunny['z'], s=1, c='blue', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Bunny Data')

plt.show()