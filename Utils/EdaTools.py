#! envs/musical-dollop/bin/python3.6
"""
EdaTools.py

Tools for exploritory data analysis of FEA data

@author: martinventer
@date: 2019-09-023

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from Utils.Utils import nodes_to_disp_array, nodes_to_coord_array, \
    elements_to_stress_array, elements_to_strain_array


def plot_3d(data) -> None:
    """plot a set of points in 3 dimensions, with consistant scale"""
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        2].flatten() + 0.5 * (z.max() + z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()
    plt.show()


def plot_disp_hist(d3plot, time_step=-1, num_bins=50):
    # load the load the displacement field to the D3plot object
    d3plot.read_states('disp')

    # Iterate over each node and store node displacement node.get_disp()
    node_displacements = nodes_to_disp_array(
        d3plot.get_nodes(), timestep=time_step
    )
    # get the displacement magnitude
    magnitude = np.linalg.norm(node_displacements, axis=1)

    # unpack and split the displacements
    u, v, w = node_displacements.T

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Displacement distribution')

    axes[0, 0].hist(u, bins=num_bins)
    axes[0, 0].set_title('x-displacement')
    # axes[0, 0].set_ylim([0, len(node_displacements)])

    axes[0, 1].hist(v, bins=num_bins)
    axes[0, 1].set_title('y-displacement')
    # axes[0, 1].set_ylim([0, len(node_displacements)])

    axes[0, 2].hist(w, bins=num_bins)
    axes[0, 2].set_title('z-displacement')
    # axes[0, 2].set_ylim([0, len(node_displacements)])

    axes[1, 1].hist(magnitude, bins=num_bins)
    axes[1, 1].set_title('displacement magnitude')
    # axes[1, 1].set_ylim([0, len(node_displacements)])


def plot_disp_quiver(d3plot, node_list=None, time_step=-1):
    # load the load the displacement field to the D3plot object
    d3plot.read_states('disp')

    if not node_list:
        node_list = d3plot.get_nodes()

    # Get node coords
    node_coords = nodes_to_coord_array(
        d3plot.get_nodeByID(node_list), timestep=time_step
    )
    x, y, z = node_coords.T

    # Get node displacements
    node_displacements = nodes_to_disp_array(
        d3plot.get_nodeByID(node_list), timestep=time_step
    )
    u, v, w = node_displacements.T

    magnitude = np.linalg.norm(node_displacements, axis=1)

    norm = Normalize()
    norm.autoscale(magnitude)

    # plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
        2].flatten() + 0.5 * (z.max() + z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.quiver(
        x, y, z, u, v, w
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_stress_hist(d3plot, time_step=-1, num_bins=50):
    # load the load the stress tensors to the D3plot object
    d3plot.read_states('stress')

    # Iterate over each element and store element stress
    element_stress_tensor = elements_to_stress_array(
        d3plot.get_elements(), timestep=time_step
    )

    # get the displacement magnitude
    magnitude = np.linalg.norm(element_stress_tensor, axis=1)

    # unpack and split the displacements
    s_xx, s_yy, s_zz, s_xy, s_yz, s_xz = element_stress_tensor.T

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Stress distribution')

    axes[0, 0].hist(s_xx, bins=num_bins)
    axes[0, 0].set_title('xx-stress')

    axes[0, 1].hist(s_yy, bins=num_bins)
    axes[0, 1].set_title('yy-stress')

    axes[0, 2].hist(s_zz, bins=num_bins)
    axes[0, 2].set_title('zz-stress')

    axes[1, 0].hist(s_xy, bins=num_bins)
    axes[1, 0].set_title('xy-stress')

    axes[1, 1].hist(s_yz, bins=num_bins)
    axes[1, 1].set_title('yz-stress')

    axes[1, 2].hist(s_xz, bins=num_bins)
    axes[1, 2].set_title('xz-stress')

    axes[2, 1].hist(magnitude, bins=num_bins)
    axes[2, 1].set_title('stress magnitude')


def plot_strain_hist(d3plot, time_step=-1, num_bins=50):
    # load the load the strain tensors to the D3plot object
    d3plot.read_states('strain')

    # Iterate over each element and store element strain
    element_strain_tensor = elements_to_strain_array(
        d3plot.get_elements(), timestep=time_step
    )

    # get the displacement magnitude
    magnitude = np.linalg.norm(element_strain_tensor, axis=1)

    # unpack and split the displacements
    e_xx, e_yy, e_zz, e_xy, e_yz, e_xz = element_strain_tensor.T

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Strain distribution')

    axes[0, 0].hist(e_xx, bins=num_bins)
    axes[0, 0].set_title('xx-Strain')

    axes[0, 1].hist(e_yy, bins=num_bins)
    axes[0, 1].set_title('yy-Strain')

    axes[0, 2].hist(e_zz, bins=num_bins)
    axes[0, 2].set_title('zz-Strain')

    axes[1, 0].hist(e_xy, bins=num_bins)
    axes[1, 0].set_title('xy-Strain')

    axes[1, 1].hist(e_yz, bins=num_bins)
    axes[1, 1].set_title('yz-Strain')

    axes[1, 2].hist(e_xz, bins=num_bins)
    axes[1, 2].set_title('xz-Strain')

    axes[2, 1].hist(magnitude, bins=num_bins)
    axes[2, 1].set_title('Strain magnitude')