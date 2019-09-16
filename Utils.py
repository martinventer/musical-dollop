#! envs/musical-dollop/bin/python3.6
"""
Utils.py

Additional utilitis required for processing of dyna results

@author: martinventer
@date: 2019-09-015

"""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm


def element_gen_node_id_list(mesh) -> list:
    for element in mesh.get_elements():
        nodes = element.get_nodes()
        yield [node.get_id() for node in nodes]


def cooc_dict(mesh) -> dict:

    num_elements = mesh.get_nElements()
    gen = element_gen_node_id_list(mesh)

    print("start")
    # create a list of all possible paris of terms
    combs = []
    for element in tqdm(gen, total=num_elements):
        possible_pairs = list(itertools.combinations(element, 2))
        combs.append(possible_pairs)

    # print("init dict")
    # # initialise a dictionary containing an entry for each pair of words
    # default_count = 0
    # cooccurring = dict.fromkeys(possible_pairs, default_count)
    #
    # print("index dict")
    # # incriment possible pairs for each occurance
    # for observation in gen:
    #     for pair in possible_pairs:
    #         if pair[0] in observation and pair[1] in observation:
    #             cooccurring[pair] += 1
    #
    # # remove cases where the co-occurance is lest than a set mimimum
    # # cooccurring = {k: v for k, v in cooccurring.items() if
    # #                v > self.minimum_occurance}
    #
    # return cooccurring

def get_surface_nodes(mesh) -> list:
    """
    Extract the node on the surface of a mesh
        1 iterate through the mesh one element at a time and extract elements
        2 build up node adjacency matrix based on shared elements
        3 build a network from the adjacency matrix
        4 extract nodes with degree less than threshold
    :param mesh:
    :return:
    """
    return []


def plot_3d(data) -> None:
    """plot a set of points in 3 dimensions"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    plt.show()


def nodes_to_coord_array(node_list, timestep=0) -> np.array:
    """convert a list of nodes to an (n, 3) numpy array"""
    points = []
    for node in node_list:
        points.append(node.get_coords())

    return np.array(points)[:, timestep, :]


class SurfaceNodes(BaseEstimator, TransformerMixin):
    """
    A transformer that takes a list of nodes from a d3plot and returns a
    subset containing only surface nodes, as a list of node objects.
    """
    def __init__(self):
        self.all_nodes_list = []
        self.all_coords = []
        self.node_ids = []
        self.surface_points = []
        self.surface_node_ids = []

    def _get_all_coords(self) -> list:
        """
        Takes the list of all_nodes, and returns a list of co-ordinate arrays
        for each node in x, y, z space.
        :return: list - of arrays containing point coords
        """
        points = []
        for node in self.all_nodes_list:
            points.append(node.get_coords())

        return points

    def _get_surface_points(self) -> list:
        """
        given a list of cords, determine which are on the surface of the
        point cloud
        :return: list - of point cords on surface of point cloud
        """
        onets = nodes_to_coord_array( self.all_nodes_list)
        # hull = ConvexHull(self.all_coords)
        hull = ConvexHull(onets)
        surface_points = list(hull.simplices)

        return surface_points

    def _get_surface_node_ids(self) -> list:
        """
        Given a list of all the coords in an object, find the points on the
        surface of that object.
        :return: list - of node ids for the surface nodes.
        """
        node_ids = []
        for node in self.all_nodes_list:
            id = node.get_id()
            coord = list(node.get_coords())
            if coord in self.surface_points:
                node_ids.append(id)

        return node_ids

    def fit(self, all_nodes_list) -> object:
        """find the nodes on the surface of the geometry"""
        self.all_nodes_list = all_nodes_list
        self.all_coords = self._get_all_coords()
        self.surface_points = self._get_surface_points()
        self.surface_node_ids = self._get_surface_node_ids()
        return self

    def transform(self) -> list:
        return self.surface_points


if __name__ == '__main__':
    from qd.cae.dyna import D3plot

    path = "/media/martin/Stuff/research/MaterailModels/MM003/"
    file = "MM003_job01.d3plot"

    # d3plot = D3plot("/media/martin/Stuff/research/MaterailModels/MM003"
    #                 "/MM003_job01.d3plot", read_states="disp")

    d3plot = D3plot(path + file, read_states="disp")

    # ==============================================================================
    # plot_3d
    # ==============================================================================
    if False:

        plot_3d(nodes_to_coord_array(d3plot.get_nodes()))

    # ==============================================================================
    # nodes_to_coord_array
    # ==============================================================================
    if False:
        coords = nodes_to_coord_array(d3plot.get_nodes())

        print(coords.shape)

    # ==============================================================================
    # element_gen_node_id_list
    # ==============================================================================
    if False:
        gen = element_gen_node_id_list(d3plot)

        print(next(gen))

    # ==============================================================================
    # cooc_dict
    # ==============================================================================
    if True:
        # gen = element_gen_node_id_list(d3plot)
        coocurrance = cooc_dict(d3plot)
        print(coocurrance)

    # ==============================================================================
    # SurfaceNodes
    # ==============================================================================
    if False:
        # surface_node_extractor = SurfaceNodes()
        # node_list = d3plot.get_nodes()
        # points = surface_node_extractor.fit_transform(
        #     node_list)
        #
        # plot_3d(points)

        node_list = d3plot.get_nodes()

        cords =[]
        for node in node_list:
            crd = node.get_coords()
            # print(crd.shape)
            cords.append(crd)

        print(len(cords))
        cords_arr = np.array(cords)
        print(cords_arr.shape)

        ts0 = cords_arr[:, 21, :]
        print(ts0.shape)
        plot_3d(ts0)


        hull = ConvexHull(ts0)
        cords_edge_index = hull.vertices
        print(cords_edge_index.shape)
        cords_edge = ts0[cords_edge_index, :]
        plot_3d(cords_edge)

        from scipy.spatial import Delaunay

        tri = Delaunay(ts0)
        pts = tri.simplices



        pts = np.unique(pts)
        cords_surf = ts0[pts, :]
        plot_3d(cords_surf)





        # cords_surface_index = np.unique(hull.simplices)
        # print(cords_surface_index.shape)

        # cords_surface_index = list(set([i for i in cords_surface_index]))
        # print(cords_surface_index.shape)

        # cords_surf = cords_arr[cords_surface_index, 0, :]
        # print(cords_surf.shape)
        # plot_3d(cords_surf)


