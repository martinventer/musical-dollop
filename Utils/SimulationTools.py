import os
import subprocess
from datetime import date, datetime
from string import Template

import numpy as np
from scipy.spatial import distance

from qd.cae.KeyFile import KeyFile
from qd.cae.dyna import D3plot


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
        self.make_sim_folder()

        # set remaining paths and names
        self.cfile_name = '{}.msg.cfile'.format(self.sim_name)
        self.keyword_geom_name = "{}_geom.k".format(self.sim_name)
        self.keyword_main_name = "{}_main.k".format(self.sim_name)
        self.job_number = 1
        self.d3plot_name = "{}_job_{}.d3plot".format(
            self.sim_name, self.job_number)

        self.cfile = os.path.join(self.sim_wd, self.cfile_name)
        self.keyword_geom_path = os.path.join(
            self.sim_wd, self.keyword_geom_name)
        self.keyword_main_path = os.path.join(
            self.sim_wd, self.keyword_main_name)
        self.d3plot_path = os.path.join(
            self.sim_wd, self.d3plot_name)

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
        self.keyword_geom = None
        self.surface_nodes = None

        self.d3plot = None

    def make_sim_folder(self) -> None:
        if not os.path.exists(self.sim_wd):
            os.makedirs(self.sim_wd)

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
            "keyword_file_name": self.keyword_geom_path
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
            "include_file": self.keyword_geom_path
        }
        with open(self.keyword_main_template, 'r') as f:
            data = f.read()
        # f.closed

        data = self.replace(data, match)

        with open(self.keyword_main_path, "w") as text_file:
            print(data, file=text_file)

    def run_simulation(self) -> str:
        print("start sim")
        d = dict(os.environ)
        d["LSTC_LICENSE"] = 'network'
        d["LSTC_LICENSE_SERVER"] = '31010@license.sun.ac.za'

        bash_command = [
            self.dyna_s_path,
            "i={}".format(self.keyword_main_path),
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

    def import_geometry_keyword(self) -> None:
        """
        Imports the geometry keyword file
        Returns
        -------
            None
        """
        self.keyword_geom = KeyFile(self.keyword_geom_path, parse_mesh=True)

    def import_d3plot(self, state='disp') -> None:
        """
        Imports d3plot data file
        Returns
        -------
            None
        """
        self.d3plot = D3plot(self.d3plot_path, read_states=state)

    def get_puck_surface_nodes(self, epsilon=0.00001) -> list:
        """
        Collects the node IDs for nodes on the surface of the puck
        Parameters
        ----------
        epsilon: float the tolerance distance from the radius curve

        Returns
        -------
            None
        """
        threshold_dist = self.radius - epsilon
        self.surface_nodes = []
        for node in self.keyword_geom.get_nodes():
            _, y, z = node.get_coords()[0]
            a = (y, z)
            b = (0, 0)
            dst = distance.euclidean(a, b)
            if dst > threshold_dist:
                self.surface_nodes.append(node.get_id())

        return self.surface_nodes