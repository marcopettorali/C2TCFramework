from matplotlib import rcParams

_IS_LATEX_INITIALIZED = False

MARKERS = ["s", "o", "v", "*", "p", "<", "h"]
COLORS = ["k", "b", "g", "r", "c", "m", "y", "maroon", "limegreen"]
HATCHES = ["", "/", "x", "\\", "o"]
STYLES = ["-", "--", "-.", ":"]

PALETTES = {"greenblue4": ["green", "limegreen", "cornflowerblue", "blue"]}

LEGEND_SMALL_SIZE = 12


def latex_initialize():
    global _IS_LATEX_INITIALIZED

    rcParams["ps.useafm"] = True
    rcParams["pdf.use14corefonts"] = True
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Helvetica"
    rcParams["font.size"] = 17.0
    rcParams["hatch.linewidth"] = 0.4
    rcParams["xtick.labelsize"] = 16
    rcParams["ytick.labelsize"] = 16
    rcParams["legend.fontsize"] = 16
    rcParams["hatch.linewidth"] = 0.7
    rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"

    _IS_LATEX_INITIALIZED = True


def latex_set_tick_fontisize(fontsize):
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize


def bold(s):
    if _IS_LATEX_INITIALIZED == False:
        return s

    if isinstance(s, str):
        # wrap every substring limited with $ ... $ with \mathbf{...}

        # find all substrings limited with $ ... $

        start = 0
        while True:
            start = s.find("$", start)
            if start == -1:
                break
            end = s.find("$", start + 1)
            if end == -1:
                break
            s = s[:start] + "_PLACEHOLDER_\\boldsymbol{" + s[start + 1 : end] + "}_PLACEHOLDER_" + s[end + 1 :]
            start = end + 1

        s = s.replace("_PLACEHOLDER_", "$")
        return "\\textbf{" + s + "}"
    elif isinstance(s, int):
        return "\\textbf{" + str(s) + "}"
    elif isinstance(s, float):
        return "\\textbf{" + str(s) + "}"
    else:
        list = []
        for item in s:
            list.append(bold(item))
        return list


# convert the matrix to latex
def matrix_to_latex_table(matrix, xs, ys, xlabel, ylabel, caption, label, doc=False, tab=False):
    latex = ""

    if doc:
        latex += "\\documentclass{article}\n"
        latex += "\\usepackage[utf8]{inputenc}\n"
        latex += "\\title{test}\n"
        latex += "\\begin{document}\n"

    if tab:
        latex += "\\begin{table}[]\n"
        latex += "\\centering\n"

    latex += "\\begin{tabular}{c |"
    for i in range(len(matrix[0])):
        latex += " c"
    latex += "}\n"

    # put multicolumn as large as the row
    latex += "& \\multicolumn{" + str(len(matrix[0])) + "}{c}{" + ylabel + "} \\\\ \n"

    latex += f"{xlabel} &"
    for i in range(len(matrix[0])):
        # check if ys[i] is a number
        if isinstance(ys[i], str):
            latex += f"{ys[i]}"
        else:
            latex += f"{round(ys[i],2)}"
        if i != len(matrix[0]) - 1:
            latex += " &"
    latex += "\\\\\n"
    latex += "\\hline\n"
    for i, row in enumerate(matrix):
        # check if xs[i] is a number
        if isinstance(xs[i], str):
            latex += f"{xs[i]}&"
        else:
            latex += f"{round(xs[i],2)}&"
        for j, cell in enumerate(row):
            latex += f"{cell}"
            if j < len(row) - 1:
                latex += " & "
        latex += "\\\\\n"
    latex += "\\end{tabular}\n"

    if tab:
        latex += "\\caption{" + caption + "}\n"
        latex += "\\label{tab:" + label + "}\n"
        latex += "\\end{table}\n"

    if doc:
        latex += "\\end{document}\n"

    return latex


import matplotlib.pyplot as plt
import networkx as nx

# TOPOLOGY


def draw_topology(graph):

    fig, ax = plt.subplots()

    pos = nx.multipartite_layout(graph, align="horizontal")
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, ax=ax)
    labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    return fig, ax


def draw_paths(graph, ax, src, dest):
    paths = nx.all_simple_paths(graph, source=src, target=dest)

    # draw the paths
    for path in paths:
        path_edges = []
        for i in range(len(path) - 1):
            path_edges.append((path[i], path[i + 1]))
        # draw the path
        nx.draw_networkx_edges(
            graph,
            pos=nx.multipartite_layout(graph, align="horizontal"),
            edgelist=path_edges,
            ax=ax,
            edge_color="red",
            width=2,
        )
