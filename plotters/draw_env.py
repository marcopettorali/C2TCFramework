# Copyright (C) 2025  Marco Pettorali

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import matplotlib.pyplot as plt

from networking.entities import AoI
from networking.environment import Environment
from resourceallocation.context import load_context
from utils.colors import palette
from utils.plotting import bold, latex_initialize
from utils.printing import print
import sys

latex_initialize()

parser = argparse.ArgumentParser(description="Draw the environment.")
parser.add_argument("scenario_name", type=str, help="The scenario name relative to configs/")
parser.add_argument("-o", "--output-path", type=str, default="", help="The path of the file where to save the plot relative to out/")
parser.add_argument("-s", "--show", action="store_true", help="Show the plot instead of saving it.")
parser.add_argument("--format", type=str, default="pdf", help="The format of the plot. Can be pdf, png, svg, etc.")
args = parser.parse_args()

CONFIG_FILE = f"configs/{args.scenario_name}.json"

context = load_context(CONFIG_FILE)

colors=None
offsets=None

fig, ax = plt.subplots()
context.draw(ax, offsets=offsets, colors=colors)

plt.tight_layout()

if args.show:
    plt.show()
else:
    # Combine the out folder and the output path
    # create a full path using out/ and output_path relative to out/
    import pathlib
    out_folder = pathlib.Path("out/") 
    output_path = out_folder / args.output_path 
    if not output_path.exists():
        print(f"Creating directory {output_path.parent}")
        output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(f"{output_path}/{args.scenario_name}_env.{args.format}", bbox_inches="tight")
