import numpy as np
import matplotlib.pyplot as plt

    # Generate random colors
    # if not mesh.visual.vertex_colors.any(): mesh.visual.vertex_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=dtype)
    # if not mesh.visual.face_colors.any(): mesh.visual.face_colors = np.random.randint(0, 255, (len(mesh.vertices), 4), dtype=dtype)

def viz_3d(arr):
    # Gets a numpy 3D array and plots it
    x, y, z = np.indices((256, 256, 3))
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = arr.flatten()  # Values at each point, could be used for coloring

    # Creating the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(x, y, z, c=values, cmap='viridis', marker='o')

    # Add a color bar to show the value scale
    plt.colorbar(sc, ax=ax, label='Value')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('3D Scatter Plot of Random Values')

    # Show the plot
    plt.show()
