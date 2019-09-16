from qd.cae.dyna import D3plot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import holoviews as hv
# from holoviews import dim, opts
#
# hv.extension('matplotlib')

# ==============================================================================
# D3plot basics
# ==============================================================================
if False:
    # load just the geometry
    d3plot = D3plot("TMP/d3plot", read_states="disp")

    # iterate over nodes
    for node in d3plot.get_nodes():
        coords = node.get_coords()

    # Looking at internal data that was not initially loaded
    d3plot = D3plot("TMP/d3plot")
    node = d3plot.get_nodeByID(1)
    len(node.get_disp())

    # Read displacements
    d3plot.read_states("disp")
    len(node.get_disp())

    # Read plastic strain
    d3plot.read_states("plastic_strain")

    # read Strain
    d3plot.read_states("strain")

    # Plot  to html
    # part.plot(iTimestep=0, export_filepath="model.html")

    # plottng to browser
    part = d3plot.get_partByID(1)
    part.plot(iTimestep=0)

    d3plot.plot(iTimestep=-1)

    d3plot.plot(iTimestep=0,
                element_result="strain",
                fringe_bounds=[0, 0.025])

# ==============================================================================
# Try on my own data
# ==============================================================================
if True:
    # dir = "/media/martin/Stuff/research/MaterailModels/MM003/"
    # file = "MM003_job02"

    d3plot = D3plot("/media/martin/Stuff/research/MaterailModels/MM003"
                    "/MM003_job01.d3plot", read_states="disp")

# -----------print out the file info ---------------------------------
#     print(
#         "{} nodes \n"
#         "{} elements \n"
#         "{} time steps \n".format(
#             d3plot.get_nNodes(),
#             d3plot.get_nElements(),
#             d3plot.get_nTimesteps()
#         )
#     )

    d3plot.info()

    cords = []
    for node in d3plot.get_nodes():
        cords.append(node.get_coords())

    initial_shape = []
    for node in cords:
        initial_shape.append(node[-1])

    df = pd.DataFrame(initial_shape, columns=['x', 'y', 'z'])

    # threedee = plt.figure().gca(projection='3d')
    # threedee.scatter(df.x, df.y, df.z)
    # threedee.set_xlabel('X')
    # threedee.set_ylabel('Y')
    # threedee.set_zlabel('Z')
    # plt.show()


    # from scipy.spatial import ConvexHull
    # hull = ConvexHull(initial_shape)
    #
    # len(hull.simplices)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.x, df.y, df.z)

plt.show()

