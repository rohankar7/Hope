# import numpy as np
# import matplotlib.pyplot as plt

#     # Generate random colors
#     # if not mesh.visual.vertex_colors.any(): mesh.visual.vertex_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=dtype)
#     # if not mesh.visual.face_colors.any(): mesh.visual.face_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=dtype)

# def viz_3d(arr):
#     # Gets a numpy 3D array and plots it
#     x, y, z = np.indices((128, 128, 3), (128, 128, 3), (128, 128, 3))
#     x = x.flatten()
#     y = y.flatten()
#     z = z.flatten()
#     values = arr.flatten()  # Values at each point, could be used for coloring

#     # Creating the plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Scatter plot
#     sc = ax.scatter(x, y, z, c=values, cmap='viridis', marker='o')

#     # Add a color bar to show the value scale
#     plt.colorbar(sc, ax=ax, label='Value')

#     # Label the axes
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Set title
#     ax.set_title('3D Scatter Plot of Random Values')

#     # Show the plot
#     plt.show()
# viz_3d(np.load('./triplane_images_128/02691156_1d4ff34cdf90d6f9aa2d78d1b8d0b45c.npy'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'images' is your numpy array of shape (3, 256, 256, 3)
# Simulating a batch of 3 random images
# images = np.load('./triplane_images_128/02691156_1d4ff34cdf90d6f9aa2d78d1b8d0b45c.npy')
images = np.load('./triplane_images_256/02691156_3baa3ca477d17e1a61f1ef59130c405d.npy')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinates setup
x = np.linspace(0, 255, 256)
y = np.linspace(0, 255, 256)
X, Y = np.meshgrid(x, y)
Z = np.full((256, 256), 128)  # Middle of the range if your axis goes from 0 to 255

# XY plane at Z=128
ax.plot_surface(X, Y, Z, facecolors=images[0], rstride=1, cstride=1, shade=False)

# YZ plane at X=128
# Rotating the image for correct orientation
image_rotated = np.rot90(images[1])
ax.plot_surface(Z, X, Y, facecolors=image_rotated, rstride=1, cstride=1, shade=False)

# XZ plane at Y=128
# Rotating the image for correct orientation
image_rotated = np.rot90(images[2])
ax.plot_surface(X, Z, Y, facecolors=image_rotated, rstride=1, cstride=1, shade=False)

# Setting the limits of the plot
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)

# Labels and titles
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Orthogonal RGB Slices in 3D Space')
plt.axis(False)

plt.show()