import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math
import os


def global_extrema(dir_name):
    """Return the global maxima and minima of Ir191Di___191Ir_DNA1 and
       Event_length in all the files under the given directory."""
    # The following helper function returns the relavant maxima and minima
    # for a specific file
    def local_extrema(filename):
        data = pd.read_csv(os.path.join(dir_name, filename))
        ir191 = data["Ir191Di___191Ir_DNA1"]
        el = data["Event_length"]
        return (ir191.max(), ir191.min(), el.max(), el.min())
    ir191_max, el_max = -math.inf, -math.inf
    ir191_min, el_min = math.inf, math.inf
    # find the global maxima and minima for the two parameters across
    # all the files in the directory
    for file in os.listdir(dir_name):
        ir191_max_l, ir191_min_l, el_max_l, el_min_l = local_extrema(file)
        ir191_max = max(ir191_max, ir191_max_l)
        ir191_min = min(ir191_min, ir191_min_l)
        el_max = max(el_max, el_max_l)
        el_min = min(el_min, el_min_l)
    return (ir191_max, ir191_min, el_max, el_min)


def axes_convert(df_numpy):
    """Convert the axes according to the following rules:
        ir191: int(ir191 * 10000 // 622)
        el: int(el - 10)"""
    for idx in range(df_numpy.shape[0]):
        ir191, el, gate1 = df_numpy[idx]
        ir191 = ir191 * 10000 // 622
        el = el - 10
        gate1 = gate1
        df_numpy[idx] = np.array((ir191, el, gate1))
    return df_numpy


def convert_to_image(dir_name, filename, mode):
    """Read a data file and output a shrinked greyscale image for the dataset.
       The output image is a numpy array of shape (166, 166, 1).
       modes:
       visual - to visualise the gate
       train - to produce an image for training
       label - to produce a label for the dataset
       """
    image = np.zeros((166, 166, 1))
    data = pd.read_csv(os.path.join(dir_name, filename))[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
    data = axes_convert(data.to_numpy())
    # for each cell in the dataset, we convert the axes by the rules described above
    for cell in data:
        ir191, el, gate1 = cell
        # need to make sure these variables are integers
        ir191, el, gate1 = int(ir191), int(el), int(gate1)
        if mode == "visual":
            if gate1 == 0:
                image[ir191, el, 0] = 128
            elif gate1 == 1:
                image[ir191, el, 0] = 255
        elif mode == "train":
            image[ir191, el, 0] += 1
        elif mode == "label":
            if gate1 == 1:
                image[ir191, el, 0] = 1
    return image


### The following function is not used in this version. ###
def convex_hull_vertices(dir_name, filename):
    """Read a data file and return a list of 10 vertices that define a polygon
       approximating the convex hull (heptagon) of the in-gate data points."""
    # The following code segment computes the convex hull
    data = pd.read_csv(os.path.join(dir_name, filename))[["Ir191Di___191Ir_DNA1", "Event_length", "gate1_ir"]]
    in_gate1 = data[data["gate1_ir"] == 1]
    in_gate1 = in_gate1[["Ir191Di___191Ir_DNA1", "Event_length"]].to_numpy()
    hull = ConvexHull(in_gate1)
    # The following code segment extracts the vertices of the convex hull
    vertices = []
    for vertex in hull.vertices:
        vertices.append(tuple(in_gate1[vertex]))
    # The following code segment reduces the number of vertices to 10
    vertex_slope_diff = {}
    num_vertices = len(vertices)
    for j in range(num_vertices):
        i, k = (j - 1) % num_vertices, (j + 1) % num_vertices
        slope1 = (vertices[j][1] - vertices[i][1]) / (vertices[j][0] - vertices[i][0])
        slope2 = (vertices[k][1] - vertices[j][1]) / (vertices[k][0] - vertices[j][0])
        vertex_slope_diff = dict(sorted(vertex_slope_diff.items()))
        reduced_vertices_idx = list(vertex_slope_diff.values())[-10:]
        reduced_vertices = []
        for idx in reduced_vertices_idx:
            reduced_vertices.append(vertices[idx])
    return reduced_vertices