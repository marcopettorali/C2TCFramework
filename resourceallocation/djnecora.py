from utils.plotting import draw_paths, draw_topology
from resourceallocation.jnecora import JNecora
from utils.logging import error, debug, info, warning


class DJNecora(JNecora):
    def __init__(self, splitting_policy, selection_policy):

        if splitting_policy not in ["no_splitting", "lazy_splitting", "greedy_splitting"]:
            raise ValueError(f"Invalid splitting policy: {splitting_policy}")

        if selection_policy not in ["first_fit", "next_fit", "best_fit", "worst_fit", "random"]:
            raise ValueError(f"Invalid selection policy: {selection_policy}")

        super().__init__()

    @staticmethod
    def load_context_from_file(config_path: str, pickle_context: bool = True, pickle_folder_relative_path: str = "pickles/djnecora"):
        JNecora.load_context_from_file(config_path, pickle_context, pickle_folder_relative_path)


if __name__ == "__main__":
    import itertools
    import sys
    import argparse
    from argparse import RawTextHelpFormatter
    import matplotlib.pyplot as plt

    # with argparse, the first parameter is the scenario name relative to configs/, there is a argument "result-path" that is the path of the json file where to save the results
    # there is a command --draw-route src dest that draws the path from src to dest and exits
    parser = argparse.ArgumentParser(
        description='DJ-NECORA: Dynamic resource allocation for C2TC (2024, Marco Pettorali)\nM. Pettorali, F. Righetti, C. Vallati, S. K. Das and G. Anastasi, "Dynamic Resource Allocation in Cloud-to-Things Continuum for Real-Time IoT Applications," 2025 IEEE International Conference on Smart Computing (SMARTCOMP), Cork, Ireland, 2025, pp. 432-437, doi: 10.1109/SMARTCOMP65954.2025.00107.\nhttps://ieeexplore.ieee.org/document/11058665',
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("scenario_name", type=str, help="The scenario name relative to configs/")
    parser.add_argument("splitting_policy", type=str, help="The splitting policy to use: no_splitting, lazy_splitting, greedy_splitting")
    parser.add_argument("selection_policy", type=str, help="The selection policy to use: first_fit, next_fit, best_fit, worst_fit, random")
    parser.add_argument(
        "--result-path", type=str, default="results.json", help="The path of the json file where to save the results relative to out/"
    )
    parser.add_argument(
        "--draw-route",
        type=str,
        nargs=2,
        help="Draw the route from src to dest and exit",
    )
    args = parser.parse_args()
    # check if the scenario name is provided
    if args.scenario_name is None:
        error("Scenario name is required")
        sys.exit(1)
    # check if the result path is provided
    if args.result_path is None:
        error("Result path is required")
        sys.exit(1)
    # check if the draw path is provided
    if args.draw_route is not None:
        src, dest = args.draw_route
        # check if src and dest are in the topology graph
        context = JNecora.load_context_from_file(f"configs/{args.scenario_name}.json")
        if src not in context.topology_graph.nodes:
            error(f"Node {src} is not in the topology graph")
            sys.exit(1)
        if dest not in context.topology_graph.nodes:
            error(f"Node {dest} is not in the topology graph")
            sys.exit(1)
        # draw the path from src to dest
        fig, ax = draw_topology(context.topology_graph)
        draw_paths(context.topology_graph, ax, src, dest)
        plt.show()
        sys.exit(0)

    # Load the context from the config file
    context = DJNecora.load_context_from_file(f"configs/{args.scenario_name}.json")

    jnecora = DJNecora(args.splitting_policy, args.selection_policy)
    jnecora.set_context(context)
