#! envs/musical-dollop/bin/python3.6
"""
Utils.py

Additional utilitis required for processing of dyna results

@author: martinventer
@date: 2019-09-015

"""
import numpy as np
import networkx as nx
import os
import subprocess

from string import Template
from datetime import date, datetime

from sklearn.base import BaseEstimator, TransformerMixin

import itertools

from qd.cae.dyna import Element


def nodes_to_coord_array(node_list, timestep=0) -> np.array:
    """Extract an (n, 3) numpy array containing nodal co-ordinates from a
    list of nodes at a given timestep"""
    points = []
    for node in node_list:
        points.append(node.get_coords())

    return np.array(points)[:, timestep, :]


def nodes_to_disp_array(node_list, timestep=0) -> np.array:
    """Extract an (n, 3) numpy array containing nodal displacements from a
    list of nodes at a given timestep"""
    points = []
    for node in node_list:
        points.append(node.get_disp())

    return np.array(points)[:, timestep, :]


def elements_to_stress_array(element_list, timestep=0) -> np.array:
    """Extract a numpy array containing element stress from a
    list of nodes at a given timestep"""
    points = []
    for element in element_list:
        points.append(element.get_stress())

    return np.array(points)[:, timestep, :]


def elements_to_strain_array(element_list, timestep=0) -> np.array:
    """Extract a numpy array containing element strain from a
    list of nodes at a given timestep"""
    points = []
    for element in element_list:
        points.append(element.get_strain())

    return np.array(points)[:, timestep, :]


def elements_to_stress_vector(element_list, timestep=0) -> np.array:
    """Extract a numpy array containing element stress from a
    list of nodes at a given timestep"""
    points = []
    for element in element_list:
        points.append(element.get_stress_mises())

    return np.array(points)


def node_id_list_generator_by_element(mesh) -> list:
    """generator for element node lists"""
    for element in mesh.get_elements():
        nodes = element.get_nodes()
        yield [node.get_id() for node in nodes]


def element_id_list_generator_by_node(mesh) -> list:
    """generator for node element node lists"""
    for node in mesh.get_nodes():
        elements = node.get_elements()
        yield [element.get_id() for element in elements]


class SurfaceNodeIds(BaseEstimator, TransformerMixin):
    """
    A transformer that takes a D3plot object and returns the surface node IDS
    """
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.surface_node_ids = None

    @staticmethod
    def _node_cooc_within_element(d3plot) -> tuple:
        """generator for unique node co-occurances within elements of a mesh"""
        # keep a set of pairs that have already been seen
        seen = set()
        # iterate over each element in the mesh
        for element in node_id_list_generator_by_element(d3plot):
            # find all paris of node co-occurance in that element
            possible_pairs = list(itertools.combinations(element, 2))
            # iterate over each pair of co-occurances
            for pair in possible_pairs:
                # test whether that pair has been seen before
                if pair not in seen:
                    # if not seen before append the pair and its reverse to seen
                    seen.add(pair[::-1])
                    seen.add(pair)
                    yield pair

    @staticmethod
    def _element_cooc_with_nodes(d3plot) -> tuple:
        """generator for unique element co-occurances with shared nodes"""
        # keep a set of pairs that have already been seen
        seen = set()
        # iterate over each node in the mesh
        for node in element_id_list_generator_by_node(d3plot):
            # find all pairs of element co-occurance at that node
            possible_pairs = list(itertools.combinations(node, 2))
            # iterate over each pair of co-occurances
            for pair in possible_pairs:
                # test whether that pair has been seen before
                if pair not in seen:
                    # if not seen before append the pair and its reverse to seen
                    seen.add(pair[::-1])
                    seen.add(pair)
                    yield pair

    def _node_cooc_network(self, d3plot):
        """build up a network from a set of occurance pairs"""
        pair_gen = self._node_cooc_within_element(d3plot)
        G = nx.Graph()
        for edge in pair_gen:
            G.add_edge(*edge)
        return G

    def _element_cooc_network(self, d3plot):
        """build up a network from a set of occurance pairs"""
        pair_gen = self._element_cooc_with_nodes(d3plot)
        G = nx.Graph()
        for edge in pair_gen:
            G.add_edge(*edge)
        return G

    def _get_surface_node_ids(self, d3plot) -> list:
        """
        Extract the node on the surface of a mesh
            1 iterate through the mesh one element at a time and extract
            elements
            2 build up node co-occurance list
            3 build a network from the co-occurance list
            4 extract node ids for nodes with degree less than threshold
        :param mesh: mesh object
        :return: list of low connectivity nodes
        """
        node_network = self._node_cooc_network(d3plot)
        node_degrees = nx.degree(node_network)
        nodes_from_node_network = [
            node_id for (node_id, deg) in node_degrees if deg <= self.threshold
        ]

        element_network = self._element_cooc_network(d3plot)
        element_degrees = nx.degree(element_network)
        elements_from_element_network = [
            element_id for (element_id, deg) in element_degrees
            if deg <= self.threshold
        ]

        nodes_from_element_network = []
        for element in d3plot.get_elementByID(
                Element.solid, elements_from_element_network
        ):
            nodes = element.get_nodes()
            nodes_from_element_network += [node.get_id() for node in nodes]

        return list(
            set(nodes_from_node_network) & set(nodes_from_element_network)
        )

    def fit(self, d3plot) -> object:
        self.surface_node_ids = self._get_surface_node_ids(d3plot)
        return self

    def transform(self, d3plot) -> list:
        yield from self.surface_node_ids


def extract_observed_surface(d3plot, time_step=0, eps=0.0001) -> list:
    """
    Extract of portion of the HD FEM surface nodes that will be tracked as if by
    DIC
    :param d3plot: object, qd.dyna.d3plot object
    :param time_step: int, The time step at which the extraction should occur
    :param eps: float, the dimension of the buffer region
    :return: list, node Ids on the surface of the boject
    """
    # We only need to consider the surface nodes
    extractor = SurfaceNodeIds(threshold=20)
    node_ids_surface = extractor.fit_transform(d3plot)
    nodes_surface = d3plot.get_nodeByID(
        list(node_ids_surface)
    )

    # get the co-ordinated of the surface nodes
    cords_surface = nodes_to_coord_array(nodes_surface, timestep=time_step)

    # determine the bounding coordinates for each dimension
    x_coord_range = (cords_surface[:, 0].min(), cords_surface[:, 0].max())
    y_coord_range = (cords_surface[:, 1].min(), cords_surface[:, 1].max())
    z_coord_range = (cords_surface[:, 2].min(), cords_surface[:, 2].max())

    # find the center of the node coords
    y_center = (y_coord_range[1] + y_coord_range[0]) / 2
    z_center = (z_coord_range[1] + z_coord_range[0]) / 2

    # iterate over each surface node and extract only those that are in
    #  the posictive y-z quadrant and not on the top or bottom x surfaces
    node_ids_observed = []
    for node in nodes_surface:
        x_coord, y_coord, z_coord = node.get_coords()[time_step]
        # check that the noded is not on the top or bottom
        if (x_coord_range[1] -eps) > x_coord > (x_coord_range[0] + eps):
            # # check positive y-z quadrant
            if y_coord > y_center and z_coord > z_center:
                node_ids_observed.append(node.get_id())

    # return the nodes meeting this criteria
    return node_ids_observed


class PuckSim:
    """
    Generate all files required to run a new puck compression simulation dyna
    simulation
    """

    def __init__(
            self, sim_name, cfile_template_name, keyword_main_template_name,
            template_dir=None, sim_dir=None):
        """
        initialise the file and direcory names for a puck simulation
        Parameters
        ----------
        sim_name: str
        cfile_template_name: str
        keyword_main_template_name: str
        template_dir: str
        sim_dir: str
        """
        self.sim_name = sim_name
        self.cfile_template_name = cfile_template_name
        self.keyword_main_template_name = keyword_main_template_name
        self.path_root = os.getcwd()

        # set default paths
        if template_dir is None:
            self.template_dir = os.path.join(self.path_root, "templates")
        else:
            self.template_dir = template_dir

        self.cfile_template = os.path.join(
            self.template_dir, self.cfile_template_name)
        self.keyword_main_template = os.path.join(
            self.template_dir, self.keyword_main_template_name)

        if sim_dir is None:
            self.sim_dir = os.path.join(self.path_root, "TMP")
        else:
            self.sim_dir = sim_dir

        self.sim_wd = os.path.join(self.sim_dir, self.sim_name)

        # set remaining paths and names
        self.cfile_name = '{}.msg.cfile'.format(self.sim_name)
        self.keyword_geom_name = "{}_geom.k".format(self.sim_name)
        self.keyword_main_name = "{}_main.k".format(self.sim_name)
        self.job_number = 1

        self.cfile = os.path.join(self.sim_wd, self.cfile_name)
        self.keyword_geom = os.path.join(self.sim_wd, self.keyword_geom_name)
        self.keyword_main = os.path.join(self.sim_wd, self.keyword_main_name)

        # additional parameters for model generation
        self.radius = None
        self.length = None
        self.num_elements = None
        self.num = None
        self.date_created = None
        self.time_created = None
        self.title = None
        self.date_time = None

        # software paths
        self.lspp_path = "/home/martin/.local/bin/lspp47"
        self.dyna_path = "/home/martin/Programs/LSTC/LSDYNA/"
        self.dyna_s_path = os.path.join(self.dyna_path,
                                        "ls-dyna_smp_s_r1010_x64_suse11_pgi165")
        self.dyna_d_path = os.path.join(self.dyna_path,
                                        "ls-dyna_smp_d_r1010_x64_suse11_pgi165")

        self.sim_output = None
        self.sim_error = None

    def setup_simulation(self, radius, length, num_elements) -> None:
        """
        update the mesh
        Parameters
        ----------
        radius: float
        length: float
        num_elements: int

        Returns
        -------
            None
        """
        self.radius = radius
        self.length = length
        self.num_elements = num_elements
        cylinder_ratio = (self.radius * 2 * np.pi) / self.length
        self.num = int(self.num_elements / cylinder_ratio)

        self.date_created = date.today()
        self.time_created = datetime.now().strftime("%H:%M:%S")
        self.title = "{} [{}]".format(self.sim_name, self.date_created)
        self.date_time = "{} ({})".format(self.date_created, self.time_created)

        # setup all components of the simulation
        self._make_cfile()
        self._build_geom_keyword()
        self._update_main_keyword()

    def _make_cfile(self) -> None:
        """
        reads the cfile macro and updates the geometry parameters, saves the
        updated cfile in the target directory
        Parameters
        ----------

        Returns
        -------
            None
        """
        match = {
            "working_directory": self.sim_wd,
            "keyword_file_name": self.keyword_geom
        }
        with open(self.cfile_template, 'r') as f:
            data = f.read()
        # f.closed

        data = self.replace(data, match)

        with open(self.cfile, "w") as text_file:
            print(data, file=text_file)

    @staticmethod
    def replace(text, dic) -> str:
        """
        replaces text in a given file.
        Parameters
        ----------
        text
        dic

        Returns
        -------

        """
        s = Template(text)
        return s.safe_substitute(**dic)

    def _build_geom_keyword(self) -> None:
        """
        Runs the lsprepost macro and creates a new geometry keyword file
        Returns
        -------
            None
        """
        bash_command = "{} c={} -nographics".format(
            self.lspp_path,
            self.cfile)

        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    def _update_main_keyword(self):
        """
        Make updates to the main keyword file so that it includes the updated
        Name and links to other files
        Returns
        -------

        """
        match = {
            "date_time": self.date_time,
            "title": self.title,
            "include_file": self.keyword_geom
        }
        with open(self.keyword_main_template, 'r') as f:
            data = f.read()
        # f.closed

        data = self.replace(data, match)

        with open(self.keyword_main, "w") as text_file:
            print(data, file=text_file)

    def run_simulation(self) -> str:
        print("start sim")
        d = dict(os.environ)
        d["LSTC_LICENSE"] = 'network'
        d["LSTC_LICENSE_SERVER"] = '31010@license.sun.ac.za'

        bash_command = [
            self.dyna_s_path,
            "i={}".format(self.keyword_main),
            "ncpu=8",
            "memory=1000m",
            "jobid={}_job_{}".format(self.sim_name, self.job_number)
        ]

        process = subprocess.Popen(bash_command,
                                   cwd=self.sim_wd,
                                   stdout=subprocess.PIPE, env=d)
        output, error = process.communicate()
        self.sim_output = output.decode('ascii')
        self.sim_error = error
        print("end sim")
        self.job_number += 1


# self.job_name = "{}_job_{}".format(self.job_number)

# TODO create a new class for generating puck geometries
# TODO method update cfile
# TODO method run cfile to creat gemetry.k
# TODO method update main.k
# TODO method run main.k in given wd
